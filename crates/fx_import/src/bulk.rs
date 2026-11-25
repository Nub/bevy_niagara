//! Editor bulk data: the package trailer (UE 5.1+) that holds the payloads
//! of `FEditorBulkData` (texture source art, static mesh descriptions), and
//! the `FCompressedBuffer` each payload is stored as (uncompressed or
//! Oodle; Oodle is decoded with the `oozextract` crate).
//!
//! An export's native data names its payloads by their 20-byte content
//! hash; [`payloads_of`] finds the trailer entries whose hash appears there.

use crate::package::{Object, Package};
use crate::reader::{Error, Reader, Result};

/// FPackageTrailer header and footer tags.
const TRAILER_HEADER_TAG: u64 = 0xD1C4_3B2E_80A5_F697;
/// FCompressedBuffer magic (big-endian in the file).
const COMPRESSED_MAGIC: u32 = 0xB775_6362;

/// One payload in the package trailer.
#[derive(Clone, Debug, PartialEq)]
pub struct Payload {
    /// FIoHash of the payload's raw content.
    pub id: [u8; 20],
    /// Absolute file offset of its FCompressedBuffer.
    pub offset: usize,
    pub compressed_size: usize,
    pub raw_size: usize,
}

/// The payloads stored locally in the package trailer (none for packages
/// without one).
pub fn trailer(pkg: &Package, bytes: &[u8]) -> Result<Vec<Payload>> {
    let toc = pkg.summary.payload_toc_offset;
    if toc <= 0 || toc as usize >= bytes.len() {
        return Ok(Vec::new());
    }
    let mut r = Reader::new(bytes);
    r.seek(toc as usize)?;
    let tag = r.u64()?;
    if tag != TRAILER_HEADER_TAG {
        return Err(r.err(format!("package trailer tag {tag:#x}")));
    }
    let version = r.i32()?;
    let header_len = r.u32()? as usize;
    let _payloads_len = r.u64()?;
    let n = r.count(20)?;
    let data_start = toc as usize + header_len;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = r.array::<20>()?;
        let offset = r.i64()?;
        let compressed_size = r.u64()? as usize;
        let raw_size = r.u64()? as usize;
        if version >= 2 {
            let _flags = r.u16()?;
            let _filter = r.u16()?;
        } else {
            let _flags = r.u32()?;
        }
        if version >= 1 {
            let _access = r.u8()?;
        }
        if offset < 0 {
            continue; // virtualized elsewhere / referenced from another file
        }
        out.push(Payload {
            id,
            offset: data_start + offset as usize,
            compressed_size,
            raw_size,
        });
    }
    Ok(out)
}

/// The trailer payloads an export's native data refers to, in the order
/// they are named there.
pub fn payloads_of(pkg: &Package, bytes: &[u8], obj: &Object) -> Result<Vec<Payload>> {
    let all = trailer(pkg, bytes)?;
    let end = (obj.native_offset + obj.native_size).min(bytes.len());
    let native = bytes.get(obj.native_offset..end).unwrap_or(&[]);
    let mut found: Vec<(usize, Payload)> = all
        .into_iter()
        .filter_map(|p| native.windows(20).position(|w| w == p.id).map(|at| (at, p)))
        .collect();
    found.sort_by_key(|(at, _)| *at);
    Ok(found.into_iter().map(|(_, p)| p).collect())
}

/// A payload's raw bytes.
pub fn read_payload(bytes: &[u8], p: &Payload) -> Result<Vec<u8>> {
    let end = p.offset.checked_add(p.compressed_size).filter(|&e| e <= bytes.len());
    let Some(end) = end else {
        return Err(Error::new(p.offset, "payload past the end of the file"));
    };
    let out = decompress(&bytes[p.offset..end])?;
    if out.len() != p.raw_size {
        return Err(Error::new(p.offset, format!("payload is {} bytes, expected {}", out.len(), p.raw_size)));
    }
    Ok(out)
}

/// Decodes an FCompressedBuffer (header in big-endian, then the block
/// sizes, then the blocks).
pub fn decompress(buf: &[u8]) -> Result<Vec<u8>> {
    let be32 = |o: usize| buf.get(o..o + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let be64 = |o: usize| buf.get(o..o + 8).map(|b| u64::from_be_bytes(b.try_into().unwrap()));
    let bad = |m: &str| Error::new(0, m.to_string());
    if be32(0) != Some(COMPRESSED_MAGIC) {
        return Err(bad("not a compressed buffer (magic)"));
    }
    let method = buf[8];
    let block_exp = buf[11] as u32;
    let blocks = be32(12).ok_or_else(|| bad("short header"))? as usize;
    let raw = be64(16).ok_or_else(|| bad("short header"))? as usize;
    const HEADER: usize = 64;
    match method {
        0 => buf.get(HEADER..HEADER + raw).map(|b| b.to_vec()).ok_or_else(|| bad("short uncompressed buffer")),
        3 => {
            let block_size = 1usize << block_exp;
            let mut sizes = Vec::with_capacity(blocks);
            for i in 0..blocks {
                sizes.push(be32(HEADER + 4 * i).ok_or_else(|| bad("short block table"))? as usize);
            }
            let mut out = vec![0u8; raw];
            let mut at = HEADER + 4 * blocks;
            let mut ex = oozextract::Extractor::new();
            for (i, &cs) in sizes.iter().enumerate() {
                let o0 = i * block_size;
                let o1 = (o0 + block_size).min(raw);
                let src = buf.get(at..at + cs).ok_or_else(|| bad("block past the end"))?;
                if cs == o1 - o0 {
                    out[o0..o1].copy_from_slice(src);
                } else {
                    let n = ex
                        .read_from_slice(src, &mut out[o0..o1])
                        .map_err(|e| bad(&format!("oodle block {i}: {e:?}")))?;
                    if n != o1 - o0 {
                        return Err(bad(&format!("oodle block {i}: {n} of {} bytes", o1 - o0)));
                    }
                }
                at += cs;
            }
            Ok(out)
        }
        m => Err(bad(&format!("compression method {m} not supported (0 none, 3 Oodle)"))),
    }
}

// ---- UE4 bulk data ----

/// FUntypedBulkData flags (UE4).
const BULKDATA_PAYLOAD_AT_END_OF_FILE: u32 = 0x1;
const BULKDATA_SERIALIZE_COMPRESSED_ZLIB: u32 = 0x2;
const BULKDATA_UNUSED: u32 = 0x20;
const BULKDATA_PAYLOAD_IN_SEPARATE_FILE: u32 = 0x100;
const BULKDATA_SIZE_64BIT: u32 = 0x2000;
const BULKDATA_NO_OFFSET_FIX_UP: u32 = 0x1_0000;
/// FArchive::SerializeCompressed chunk header tag (the package file tag).
const ZLIB_CHUNK_TAG: i64 = crate::summary::PACKAGE_FILE_TAG as i64;

/// The UE4 texture source art of a Texture2D export (packages without a
/// trailer, legacy file version -7): its native data starts with the
/// UTexture strip flags, then the source's FByteBulkData. The payload sits
/// inline or at the end of the file (`BulkDataStartOffset` + offset), and
/// may be zlib-compressed in FArchive::SerializeCompressed chunks.
pub fn ue4_texture_source(pkg: &Package, bytes: &[u8], obj: &Object) -> Result<Vec<u8>> {
    let end = (obj.native_offset + obj.native_size).min(bytes.len());
    let mut r = Reader::new(bytes).window(obj.native_offset, end)?;
    r.bytes(2)?; // FStripDataFlags (global, class)
    let flags = r.u32()?;
    let (count, on_disk) = if flags & BULKDATA_SIZE_64BIT != 0 {
        (r.i64()?, r.i64()?)
    } else {
        (r.i32()? as i64, r.i32()? as i64)
    };
    let offset = r.i64()?;
    if flags & BULKDATA_UNUSED != 0 || count == 0 {
        return Err(r.err("texture source bulk data is empty"));
    }
    if flags & BULKDATA_PAYLOAD_IN_SEPARATE_FILE != 0 {
        return Err(r.err("texture source bulk data is in a separate file"));
    }
    let start = if flags & BULKDATA_PAYLOAD_AT_END_OF_FILE != 0 {
        let fix = if flags & BULKDATA_NO_OFFSET_FIX_UP != 0 { 0 } else { pkg.summary.bulk_data_start_offset };
        (offset + fix) as usize
    } else {
        r.pos()
    };
    let data = start
        .checked_add(on_disk.max(0) as usize)
        .and_then(|e| bytes.get(start..e))
        .ok_or_else(|| Error::new(start, "bulk data past the end of the file"))?;
    let out = if flags & BULKDATA_SERIALIZE_COMPRESSED_ZLIB != 0 { inflate_chunks(data)? } else { data.to_vec() };
    if out.len() as i64 != count {
        return Err(Error::new(start, format!("bulk data is {} bytes, expected {count}", out.len())));
    }
    Ok(out)
}

/// FArchive::SerializeCompressed (zlib): tag {PACKAGE_FILE_TAG, chunk
/// size}, summary {compressed, uncompressed}, one {compressed,
/// uncompressed} pair per chunk (all i64), then the zlib streams.
pub fn inflate_chunks(buf: &[u8]) -> Result<Vec<u8>> {
    let mut r = Reader::new(buf);
    if r.i64()? != ZLIB_CHUNK_TAG {
        return Err(r.err("compressed bulk data: bad chunk tag"));
    }
    let chunk_size = r.i64()?.max(1);
    let _compressed = r.i64()?;
    let total = r.i64()?;
    if total < 0 || total > 1 << 32 {
        return Err(r.err("compressed bulk data: bad size"));
    }
    let n = (total + chunk_size - 1) / chunk_size;
    let mut sizes = Vec::with_capacity(n as usize);
    for _ in 0..n {
        sizes.push((r.i64()? as usize, r.i64()? as usize));
    }
    let mut out = Vec::with_capacity(total as usize);
    for (c, u) in sizes {
        let at = r.pos();
        let src = r.bytes(c)?;
        let chunk = miniz_oxide::inflate::decompress_to_vec_zlib(src)
            .map_err(|e| Error::new(at, format!("zlib: {e:?}")))?;
        if chunk.len() != u {
            return Err(Error::new(at, format!("zlib chunk is {} bytes, expected {u}", chunk.len())));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}
