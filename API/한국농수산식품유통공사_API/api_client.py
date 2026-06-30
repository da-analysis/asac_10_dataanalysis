import requests
import json
import xml.dom.minidom
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BaseAPIClient:
    def __init__(self, service_key, base_url):
        self.service_key = service_key
        self.base_url = base_url.rstrip('/')

    def _get(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.get(url, params=params, verify=False, timeout=15)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"[ERROR] API 호출 실패: {url} | {e}")
            return None

    def _parse_response(self, response, return_type):
        if not response:
            return None
        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            try:
                return xml.dom.minidom.parseString(response.content).toprettyxml()
            except:
                return response.content


class ATAPIClient(BaseAPIClient):
    """aT 한국농수산식품유통공사 API 클라이언트"""

    def __init__(self, service_key):
        super().__init__(service_key, "https://apis.data.go.kr")

    def _build_params(self, page, rows, return_type, **conditions):
        params = {
            "serviceKey": self.service_key,
            "pageNo": str(page),
            "numOfRows": str(rows),
            "returnType": return_type.upper(),
        }
        for key, val in conditions.items():
            if val is not None:
                params[key] = val
        return params

    def _call(self, endpoint, params, return_type):
        print(f"[호출] https://apis.data.go.kr/{endpoint}")
        response = self._get(endpoint, params=params)
        return self._parse_response(response, return_type)

    # ── 지역별 가격정보 (perRegion) ──
    def fetch_region_price(self, date_gte, date_lte, sgg_cd,
                           se_cd=None, ctgry_cd=None, item_cd=None,
                           vrty_cd=None, grd_cd=None, selectable=None,
                           page=1, rows=10, return_type="json"):
        params = self._build_params(page, rows, return_type,
            **{"cond[exmn_ymd::GTE]": date_gte,
               "cond[exmn_ymd::LTE]": date_lte,
               "cond[sgg_cd::EQ]": sgg_cd,
               "cond[se_cd::EQ]": se_cd,
               "cond[ctgry_cd::EQ]": ctgry_cd,
               "cond[item_cd::EQ]": item_cd,
               "cond[vrty_cd::EQ]": vrty_cd,
               "cond[grd_cd::EQ]": grd_cd,
               "selectable": selectable})
        return self._call("B552845/perRegion/price", params, return_type)

    # ── 일자별 가격정보 (perDay) ──
    def fetch_day_price(self, date_gte, date_lte, ctgry_cd, item_cd,
                        se_cd=None, vrty_cd=None, grd_cd=None,
                        sgg_cd=None, mrkt_cd=None, selectable=None,
                        page=1, rows=10, return_type="json"):
        params = self._build_params(page, rows, return_type,
            **{"cond[exmn_ymd::GTE]": date_gte,
               "cond[exmn_ymd::LTE]": date_lte,
               "cond[ctgry_cd::EQ]": ctgry_cd,
               "cond[item_cd::EQ]": item_cd,
               "cond[se_cd::EQ]": se_cd,
               "cond[vrty_cd::EQ]": vrty_cd,
               "cond[grd_cd::EQ]": grd_cd,
               "cond[sgg_cd::EQ]": sgg_cd,
               "cond[mrkt_cd::EQ]": mrkt_cd,
               "selectable": selectable})
        return self._call("B552845/perDay/price", params, return_type)

    # ── 연월별 가격정보 (perYearMonth) ──
    def fetch_month_price(self, month_gte, month_lte, sgg_cd,
                          se_cd=None, ctgry_cd=None, item_cd=None,
                          vrty_cd=None, grd_cd=None, selectable=None,
                          page=1, rows=10, return_type="json"):
        params = self._build_params(page, rows, return_type,
            **{"cond[exmn_ym::GTE]": month_gte,
               "cond[exmn_ym::LTE]": month_lte,
               "cond[sgg_cd::EQ]": sgg_cd,
               "cond[se_cd::EQ]": se_cd,
               "cond[ctgry_cd::EQ]": ctgry_cd,
               "cond[item_cd::EQ]": item_cd,
               "cond[vrty_cd::EQ]": vrty_cd,
               "cond[grd_cd::EQ]": grd_cd,
               "selectable": selectable})
        return self._call("B552845/perYearMonth/price", params, return_type)

    # ── 가격등락정보 (risesAndFalls) ──
    def fetch_price_change(self, exmn_ymd,
                           se_cd=None, ctgry_cd=None, item_cd=None,
                           vrty_cd=None, grd_cd=None, selectable=None,
                           page=1, rows=10, return_type="json"):
        params = self._build_params(page, rows, return_type,
            **{"cond[exmn_ymd::EQ]": exmn_ymd,
               "cond[se_cd::EQ]": se_cd,
               "cond[ctgry_cd::EQ]": ctgry_cd,
               "cond[item_cd::EQ]": item_cd,
               "cond[vrty_cd::EQ]": vrty_cd,
               "cond[grd_cd::EQ]": grd_cd,
               "selectable": selectable})
        return self._call("B552845/risesAndFalls/info", params, return_type)

    # ── 가격 추이정보 (priceSequel) ──
    def fetch_price_trend(self, exmn_ymd,
                          se_cd=None, ctgry_cd=None, item_cd=None,
                          vrty_cd=None, grd_cd=None, selectable=None,
                          page=1, rows=10, return_type="json"):
        params = self._build_params(page, rows, return_type,
            **{"cond[exmn_ymd::EQ]": exmn_ymd,
               "cond[se_cd::EQ]": se_cd,
               "cond[ctgry_cd::EQ]": ctgry_cd,
               "cond[item_cd::EQ]": item_cd,
               "cond[vrty_cd::EQ]": vrty_cd,
               "cond[grd_cd::EQ]": grd_cd,
               "selectable": selectable})
        return self._call("B552845/priceSequel/info", params, return_type)

    # ── 출하일 추이정보 (shipmentSequel) ──
    def fetch_shipment_trend(self, spmt_ymd,
                             whsl_mrkt_cd=None, corp_cd=None,
                             gds_lclsf_cd=None, gds_mclsf_cd=None,
                             gds_sclsf_cd=None, selectable=None,
                             page=1, rows=10, return_type="json"):
        params = self._build_params(page, rows, return_type,
            **{"cond[spmt_ymd::EQ]": spmt_ymd,
               "cond[whsl_mrkt_cd::EQ]": whsl_mrkt_cd,
               "cond[corp_cd::EQ]": corp_cd,
               "cond[gds_lclsf_cd::EQ]": gds_lclsf_cd,
               "cond[gds_mclsf_cd::EQ]": gds_mclsf_cd,
               "cond[gds_sclsf_cd::EQ]": gds_sclsf_cd,
               "selectable": selectable})
        return self._call("B552845/shipmentSequel/info", params, return_type)


class NCSWholesaleClient(BaseAPIClient):
    """농림축산식품 공공데이터포털 (도매시장 경매정보) 클라이언트"""

    def __init__(self, service_key):
        super().__init__(service_key, "http://211.237.50.150:7080/openapi")

    def fetch_auction_info(self, service_id, sale_date, whsal_cd,
                           start_idx=1, rows=10, return_type="json"):
        end_idx = start_idx + rows - 1
        endpoint = f"{self.service_key}/{return_type.lower()}/{service_id}/{start_idx}/{end_idx}"
        params = {"SALEDATE": sale_date, "WHSALCD": whsal_cd}
        response = self._get(endpoint, params=params)
        return self._parse_response(response, return_type)
