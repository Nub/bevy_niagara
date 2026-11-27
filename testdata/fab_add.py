# Adds free Fab listings to the user's Fab library, as the site's "Add to My
# Library" button does, from the fab-downloader launcher session (no browser,
# password or cookie export):
#   1. launcher token -> Epic exchange code (account API /oauth/exchange)
#   2. exchange code -> epicgames.com web session (/id/api/exchange)
#   3. fab.com Epic login (/social/login/epic), its authorize step answered by
#      /id/api/redirect -> fab_sessionid + fab_csrftoken cookies
#   4. POST /i/listings/<uid>/add-to-library offer_id=<the free licence's offer>
# Only free offers (price 0) are taken; paid listings are refused. Never prints
# tokens or cookies.
#
# Usage, from the repository root:
#   nix shell nixpkgs#python3 -c python3 -I testdata/fab_add.py LISTING_UID ...
# then `python3 -m fabdl list` (in testdata/tools/fab-downloader) shows the new
# library asset ids for fab_assets.sh.
import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "fab-downloader")
sys.path.insert(0, TOOL)
from fabdl.auth import SessionStore, load_session  # noqa: E402
from fabdl.net import HttpClient  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def go(url, data=None, headers=None, method=None):
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        r = opener.open(req, timeout=30)
        return r.status, r.geturl(), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.geturl() or url, e.read()


def cookie(name):
    return next((c.value for c in jar if c.name == name), "")


def login():
    s = load_session(HttpClient(), SessionStore.default())
    if s is None:
        sys.exit("no fab-downloader session: run `python3 -m fabdl list` once and sign in")
    st, _, b = go(
        "https://account-public-service-prod.ol.epicgames.com/account/api/oauth/exchange",
        headers={"Authorization": f"Bearer {s.access_token}"},
    )
    if st != 200:
        sys.exit(f"exchange code: HTTP {st}")
    code = json.loads(b)["code"]
    go("https://www.epicgames.com/id/api/csrf")
    st, _, _ = go(
        "https://www.epicgames.com/id/api/exchange",
        data=json.dumps({"exchangeCode": code}).encode(),
        headers={"Content-Type": "application/json", "X-XSRF-TOKEN": cookie("XSRF-TOKEN")},
        method="POST",
    )
    if st != 200:
        sys.exit(f"epicgames.com web session: HTTP {st}")
    st, url, _ = go("https://www.fab.com/social/login/epic?next=/")
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    if "client_id" not in q:
        sys.exit(f"fab login did not reach Epic authorize (HTTP {st})")
    p = {
        "clientId": q["client_id"][0],
        "responseType": "code",
        "redirectUrl": q["redirect_uri"][0],
        "state": q["state"][0],
        "scope": q.get("scope", [""])[0],
    }
    st, _, b = go(
        "https://www.epicgames.com/id/api/redirect?" + urllib.parse.urlencode(p),
        headers={"X-XSRF-TOKEN": cookie("XSRF-TOKEN")},
    )
    redirect = json.loads(b).get("redirectUrl", "") if b[:1] == b"{" else ""
    if "code=" not in redirect:
        sys.exit(f"Epic authorize for fab.com gave no code (HTTP {st}); consent may be needed in a browser")
    go(redirect)
    if not cookie("fab_sessionid") or not cookie("fab_csrftoken"):
        sys.exit("fab.com session not set")


def free_offer(uid):
    st, _, b = go(f"https://www.fab.com/i/listings/{uid}", headers={"Accept": "application/json"})
    if st != 200:
        sys.exit(f"{uid}: listing HTTP {st}")
    d = json.loads(b)
    offers = [l["offerId"] for l in d.get("licenses", []) if (l.get("priceTier") or {}).get("price") == 0]
    if not offers:
        sys.exit(f"{uid} ({d.get('title')}): no free licence, not added")
    return d.get("title"), offers[0]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__ or "usage: fab_add.py LISTING_UID ...")
    login()
    for uid in sys.argv[1:]:
        title, offer = free_offer(uid)
        st, _, b = go(
            f"https://www.fab.com/i/listings/{uid}/add-to-library",
            data=urllib.parse.urlencode({"offer_id": offer}).encode(),
            method="POST",
            headers={
                "X-CsrfToken": cookie("fab_csrftoken"),
                "Referer": f"https://www.fab.com/listings/{uid}",
                "Origin": "https://www.fab.com",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        ok = st in (200, 201, 204)
        print(f"{'added' if ok else 'FAILED'}: {title} ({uid}) HTTP {st} {'' if ok else b[:200]!r}")


main()
