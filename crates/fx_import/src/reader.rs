//! Little-endian byte reader with offsets in its errors.

use std::fmt;

/// A read error: what went wrong and the absolute file offset where.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    pub offset: usize,
    pub msg: String,
}

impl Error {
    pub fn new(offset: usize, msg: impl Into<String>) -> Self {
        Error {
            offset,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at offset {:#x}: {}", self.offset, self.msg)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Cursor over a window `[start, end)` of the package bytes. Positions are
/// absolute file offsets, so errors and sub-readers point into the file.
#[derive(Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader {
            data,
            pos: 0,
            end: data.len(),
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn remaining(&self) -> usize {
        self.end - self.pos
    }

    pub fn err(&self, msg: impl Into<String>) -> Error {
        Error::new(self.pos, msg)
    }

    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.end {
            return Err(self.err(format!("seek to {pos:#x} past end {:#x}", self.end)));
        }
        self.pos = pos;
        Ok(())
    }

    /// A reader over the next `len` bytes; this reader skips past them.
    pub fn sub(&mut self, len: usize) -> Result<Reader<'a>> {
        if len > self.remaining() {
            return Err(self.err(format!(
                "{len} bytes requested, {} left",
                self.remaining()
            )));
        }
        let r = Reader {
            data: self.data,
            pos: self.pos,
            end: self.pos + len,
        };
        self.pos += len;
        Ok(r)
    }

    /// A reader over `[start, end)` of the whole file.
    pub fn window(&self, start: usize, end: usize) -> Result<Reader<'a>> {
        if start > end || end > self.data.len() {
            return Err(self.err(format!(
                "window {start:#x}..{end:#x} outside file of {:#x} bytes",
                self.data.len()
            )));
        }
        Ok(Reader {
            data: self.data,
            pos: start,
            end,
        })
    }

    pub fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if len > self.remaining() {
            return Err(self.err(format!(
                "{len} bytes requested, {} left",
                self.remaining()
            )));
        }
        let s = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(s)
    }

    pub fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0; N];
        a.copy_from_slice(self.bytes(N)?);
        Ok(a)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    pub fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    pub fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.array()?))
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    pub fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.array()?))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    /// UBOOL: a 32-bit 0 or 1.
    pub fn bool32(&mut self) -> Result<bool> {
        let at = self.pos;
        match self.u32()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(Error::new(at, format!("bad bool value {v}"))),
        }
    }

    /// A non-negative i32 count, checked against the bytes left (each item
    /// takes at least `min_item` bytes).
    pub fn count(&mut self, min_item: usize) -> Result<usize> {
        let at = self.pos;
        let n = self.i32()?;
        if n < 0 || (n as usize).saturating_mul(min_item.max(1)) > self.remaining() {
            return Err(Error::new(
                at,
                format!("bad count {n} ({} bytes left)", self.remaining()),
            ));
        }
        Ok(n as usize)
    }

    /// FString: i32 length including the terminating NUL; negative means
    /// UTF-16 code units.
    pub fn fstring(&mut self) -> Result<String> {
        let at = self.pos;
        let len = self.i32()?;
        if len == 0 {
            return Ok(String::new());
        }
        if len > 0 {
            let b = self.bytes(len as usize)?;
            let b = b.strip_suffix(&[0]).ok_or_else(|| Error::new(at, "FString without NUL"))?;
            // Latin-1.
            Ok(b.iter().map(|&c| c as char).collect())
        } else {
            let n = len.unsigned_abs() as usize;
            let b = self.bytes(n * 2)?;
            let units: Vec<u16> = b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let units = units.strip_suffix(&[0]).ok_or_else(|| Error::new(at, "FString without NUL"))?;
            Ok(String::from_utf16_lossy(units))
        }
    }

    /// FUtf8String: i32 byte length (no NUL) then UTF-8.
    pub fn utf8_string(&mut self) -> Result<String> {
        let n = self.count(1)?;
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }

    pub fn guid(&mut self) -> Result<Guid> {
        Ok(Guid([self.u32()?, self.u32()?, self.u32()?, self.u32()?]))
    }
}

/// FGuid: four u32 words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Guid(pub [u32; 4]);

impl fmt::Display for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:08X}-{:08X}-{:08X}-{:08X}",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}
