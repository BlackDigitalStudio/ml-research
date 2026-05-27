"""EV-B probe (read-only): verify reachability + timestamp fidelity of exchange
listing-announcement APIs. No writes. Learns the JSON shapes for the puller."""
import json
import datetime as dt
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}


def get(name, url, headers=None, parse=None):
    print(f"\n==== {name} ====\n{url}")
    try:
        r = requests.get(url, headers={**UA, **(headers or {})}, timeout=25)
        print("HTTP", r.status_code, "| ct", r.headers.get("content-type", "")[:40], "| bytes", len(r.content))
        if r.status_code != 200:
            print("body[:300]:", r.text[:300]); return None
        try:
            j = r.json()
        except Exception:
            print("NOT JSON, body[:300]:", r.text[:300]); return None
        if parse:
            parse(j)
        return j
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:200]); return None


def p_upbit(j):
    print("top keys:", list(j)[:10])
    data = j.get("data", j)
    notices = (data.get("notices") if isinstance(data, dict) else None) or \
              (data.get("list") if isinstance(data, dict) else None) or \
              (data if isinstance(data, list) else None)
    if isinstance(notices, list) and notices:
        print("n:", len(notices), "| sample keys:", list(notices[0]))
        for x in notices[:4]:
            tsf = x.get("listed_at") or x.get("first_listed_at") or x.get("created_at")
            print("   ", tsf, "|", str(x.get("title"))[:90])


def p_bnc(j):
    d = j.get("data", {}) or {}
    arts = d.get("articles") or d.get("catalogs") or []
    print("data keys:", list(d)[:10], "| n_articles:", len(arts))
    for a in arts[:4]:
        rd = a.get("releaseDate")
        iso = dt.datetime.utcfromtimestamp(rd / 1000).isoformat() + "Z" if rd else None
        print("   ", rd, iso, "|", str(a.get("title"))[:90])


# 1) Upbit — try /announcements then /notices
u = get("Upbit /announcements",
        "https://api-manager.upbit.com/api/v1/announcements?os=web&page=1&per_page=10&category=trade",
        parse=p_upbit)
if u is None:
    get("Upbit /notices",
        "https://api-manager.upbit.com/api/v1/notices?os=web&page=1&per_page=10",
        parse=p_upbit)

# 2) Binance CMS — catalogId 48 = New Cryptocurrency Listing
get("Binance CMS catalog48 (New Listing)",
    "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query?catalogId=48&pageNo=1&pageSize=10",
    headers={"clienttype": "web", "lang": "en"}, parse=p_bnc)

# 3) Coinbase — exchange products (reachability proxy; announcement ts handled separately)
get("Coinbase exchange products",
    "https://api.exchange.coinbase.com/products",
    parse=lambda j: print("n_products:", len(j), "| sample:",
                          {k: j[0].get(k) for k in ("id", "status", "trading_disabled")} if j else None))

print("\nEV_B_PROBE_DONE")
