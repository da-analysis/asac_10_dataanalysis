import requests
import json
import xml.dom.minidom
import urllib3
import pandas as pd

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BaseAPIClient:
    """모든 API 클라이언트의 공통 기능을 담은 베이스 클래스"""
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
            print(f"[ERROR] API 호출 실패: {url}")
            print(f"이유: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"응답 내용: {response.text[:200]}...")
            return None

    def _prettify_xml(self, xml_content):
        """XML 결과를 읽기 좋게 정렬"""
        try:
            dom = xml.dom.minidom.parseString(xml_content)
            return dom.toprettyxml()
        except:
            return xml_content

class ATAPIClient(BaseAPIClient):
    """
    공공데이터포털 (aT 한국농수산식품유통공사) 전용 클라이언트
    API: 지역별 품목별 도,소매 가격정보 조회
    Endpoint: https://apis.data.go.kr/B552845/perRegion/price
    """
    def __init__(self, service_key):
        super().__init__(service_key, "https://apis.data.go.kr")

    def fetch_region_price(
        self,
        # ── 필수 파라미터 ──────────────────────────────────────
        date_gte,           # 조사일자 시작 (YYYYMMDD), 예: '20250101'
        date_lte,           # 조사일자 종료 (YYYYMMDD), 예: '20250131'
        sgg_cd,             # 시군구코드, 예: '1101' (서울)
        # ── 선택 파라미터 (필터) ───────────────────────────────
        se_cd=None,         # 구분코드: '01'=소매, '02'=중도매(도매)
        ctgry_cd=None,      # 부류코드, 예: '100'=식량작물, '200'=채소류
        item_cd=None,       # 품목코드, 예: '111'=쌀, '211'=배추
        vrty_cd=None,       # 품종코드, 예: '01'=일반
        grd_cd=None,        # 등급코드, 예: '04'=상품
        # ── 응답 제어 파라미터 ─────────────────────────────────
        selectable=None,    # 원하는 응답 컬럼만 선택 (쉼표 구분), 예: 'exmn_ymd,item_nm,exmn_dd_avg_prc'
        page=1,             # 페이지 번호 (기본값: 1)
        rows=10,            # 한 페이지 결과 수 (기본값: 10, 최대: 1000)
        return_type="json"  # 응답 형식: 'json' 또는 'xml'
    ):
        """
        지역별 품목별 도·소매 가격정보를 조회합니다.

        [필수]
          date_gte  : 조사일자 시작 (YYYYMMDD)
          date_lte  : 조사일자 종료 (YYYYMMDD)
          sgg_cd    : 시군구코드

        [선택 필터]
          se_cd     : 구분코드 ('01'=소매, '02'=중도매)
          ctgry_cd  : 부류코드 (예: '100'=식량작물, '200'=채소류, '400'=과실류)
          item_cd   : 품목코드 (예: '111'=쌀, '211'=배추, '411'=사과)
          vrty_cd   : 품종코드
          grd_cd    : 등급코드 (예: '04'=상품, '05'=중품)

        [응답 제어]
          selectable: 받고 싶은 컬럼만 콤마(,) 구분으로 지정
                      (예: 'exmn_ymd,item_nm,sgg_nm,exmn_dd_avg_prc')
          page      : 페이지 번호
          rows      : 결과 수 (최대 1000)
          return_type: 'json' 또는 'xml'

        [응답 필드 안내]
          exmn_ymd              : 조사일자
          se_cd / se_nm         : 구분코드/명
          ctgry_cd / ctgry_nm   : 부류코드/명
          item_cd / item_nm     : 품목코드/명
          vrty_cd / vrty_nm     : 품종코드/명
          grd_cd / grd_nm       : 등급코드/명
          sgg_cd / sgg_nm       : 시군구코드/명
          unit / unit_sz        : 단위/단위크기
          exmn_dd_min_prc       : 조사일 최저가격
          exmn_dd_avg_prc       : 조사일 평균가격
          exmn_dd_max_prc       : 조사일 최고가격
          exmn_dd_cnvs_avg_prc  : 조사일 환산 평균가격
        """
        endpoint = "B552845/perRegion/price"

        # 기본 + 필수 파라미터
        params = {
            "serviceKey": self.service_key,
            "pageNo":     str(page),
            "numOfRows":  str(rows),
            "returnType": return_type.upper(),
            "cond[exmn_ymd::GTE]": date_gte,
            "cond[exmn_ymd::LTE]": date_lte,
            "cond[sgg_cd::EQ]":    sgg_cd,
        }

        # 선택 필터 (값이 있을 때만 추가)
        if se_cd:     params["cond[se_cd::EQ]"]    = se_cd
        if ctgry_cd:  params["cond[ctgry_cd::EQ]"] = ctgry_cd
        if item_cd:   params["cond[item_cd::EQ]"]  = item_cd
        if vrty_cd:   params["cond[vrty_cd::EQ]"]  = vrty_cd
        if grd_cd:    params["cond[grd_cd::EQ]"]   = grd_cd

        # 응답 컬럼 선택 (선택 사항)
        if selectable:
            params["selectable"] = selectable

        print(f"[호출 URL] https://apis.data.go.kr/{endpoint}")
        print(f"[파라미터] {json.dumps({k: v for k, v in params.items() if k != 'serviceKey'}, ensure_ascii=False, indent=2)}")

        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)

    def fetch_day_price(
        self,
        # ── 필수 파라미터 ──────────────────────────────────────
        date_gte,           # 조사일자 시작 (YYYYMMDD), 예: '20250101'
        date_lte,           # 조사일자 종료 (YYYYMMDD), 예: '20250131'
        ctgry_cd,           # 부류코드, 예: '100'=식량작물
        item_cd,            # 품목코드, 예: '111'=쌀
        # ── 선택 파라미터 (필터) ───────────────────────────────
        se_cd=None,         # 구분코드: '01'=소매, '02'=중도매(도매)
        vrty_cd=None,       # 품종코드, 예: '01'=일반
        grd_cd=None,        # 등급코드, 예: '04'=상품
        sgg_cd=None,        # 시군구코드, 예: '1101'
        mrkt_cd=None,       # 시장코드
        # ── 응답 제어 파라미터 ─────────────────────────────────
        selectable=None,    # 원하는 응답 컬럼만 선택 (쉼표 구분)
        page=1,             # 페이지 번호 (기본값: 1)
        rows=10,            # 한 페이지 결과 수 (기본값: 10, 최대: 1000)
        return_type="json"  # 응답 형식: 'json' 또는 'xml'
    ):
        """
        일자별 가격정보를 조회합니다.

        [필수]
          date_gte  : 조사일자 시작 (YYYYMMDD)
          date_lte  : 조사일자 종료 (YYYYMMDD)
          ctgry_cd  : 부류코드
          item_cd   : 품목코드

        [선택 필터]
          se_cd     : 구분코드 ('01'=소매, '02'=중도매)
          vrty_cd   : 품종코드
          grd_cd    : 등급코드
          sgg_cd    : 시군구코드
          mrkt_cd   : 시장코드

        [응답 제어]
          selectable: 받고 싶은 컬럼만 콤마(,) 구분으로 지정
          page      : 페이지 번호
          rows      : 결과 수 (최대 1000)
          return_type: 'json' 또는 'xml'
        """
        endpoint = "B552845/perDay/price"

        # 기본 + 필수 파라미터 (명세에 정의된 필수 조건)
        params = {
            "serviceKey": self.service_key,
            "pageNo":     str(page),
            "numOfRows":  str(rows),
            "returnType": return_type.upper(),
            "cond[exmn_ymd::GTE]": date_gte,
            "cond[exmn_ymd::LTE]": date_lte,
            "cond[ctgry_cd::EQ]":  ctgry_cd,
            "cond[item_cd::EQ]":   item_cd,
        }

        # 선택 필터 (값이 있을 때만 추가)
        if se_cd:     params["cond[se_cd::EQ]"]    = se_cd
        if vrty_cd:   params["cond[vrty_cd::EQ]"]  = vrty_cd
        if grd_cd:    params["cond[grd_cd::EQ]"]   = grd_cd
        if sgg_cd:    params["cond[sgg_cd::EQ]"]   = sgg_cd
        if mrkt_cd:   params["cond[mrkt_cd::EQ]"]  = mrkt_cd

        # 응답 컬럼 선택 (선택 사항)
        if selectable:
            params["selectable"] = selectable

        print(f"[호출 URL] https://apis.data.go.kr/{endpoint}")
        print(f"[파라미터] {json.dumps({k: v for k, v in params.items() if k != 'serviceKey'}, ensure_ascii=False, indent=2)}")

        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)

    def fetch_price_change(
        self,
        # ── 필수 파라미터 ──────────────────────────────────────
        exmn_ymd,           # 조사일자 (YYYYMMDD), 예: '20250401' — EQ(일치) 조건
        # ── 선택 파라미터 (필터) ───────────────────────────────
        se_cd=None,         # 구분코드: '01'=소매, '02'=중도매(도매)
        ctgry_cd=None,      # 부류코드, 예: '100'=식량작물, '200'=채소류
        item_cd=None,       # 품목코드, 예: '111'=쌀, '211'=배추
        vrty_cd=None,       # 품종코드, 예: '01'=일반
        grd_cd=None,        # 등급코드, 예: '04'=상품
        # ── 응답 제어 파라미터 ─────────────────────────────────
        selectable=None,    # 원하는 응답 컬럼만 선택 (쉼표 구분)
        page=1,             # 페이지 번호 (기본값: 1)
        rows=10,            # 한 페이지 결과 수 (기본값: 10, 최대: 1000)
        return_type="json"  # 응답 형식: 'json' 또는 'xml'
    ):
        """
        가격등락정보를 조회합니다.

        [필수]
          exmn_ymd  : 조사일자 (YYYYMMDD) — 정확히 일치하는 날짜만 검색

        [선택 필터]
          se_cd     : 구분코드 ('01'=소매, '02'=중도매)
          ctgry_cd  : 부류코드 (예: '100'=식량작물, '200'=채소류, '400'=과실류)
          item_cd   : 품목코드 (예: '111'=쌀, '211'=배추, '411'=사과)
          vrty_cd   : 품종코드
          grd_cd    : 등급코드 (예: '04'=상품, '05'=중품)

        [응답 제어]
          selectable: 받고 싶은 컬럼만 콤마(,) 구분으로 지정
          page      : 페이지 번호
          rows      : 결과 수 (최대 1000)
          return_type: 'json' 또는 'xml'
        """
        endpoint = "B552845/risesAndFalls/info"

        # 기본 + 필수 파라미터
        params = {
            "serviceKey": self.service_key,
            "pageNo":     str(page),
            "numOfRows":  str(rows),
            "returnType": return_type.upper(),
            "cond[exmn_ymd::EQ]": exmn_ymd,
        }

        # 선택 필터 (값이 있을 때만 추가)
        if se_cd:     params["cond[se_cd::EQ]"]    = se_cd
        if ctgry_cd:  params["cond[ctgry_cd::EQ]"] = ctgry_cd
        if item_cd:   params["cond[item_cd::EQ]"]  = item_cd
        if vrty_cd:   params["cond[vrty_cd::EQ]"]  = vrty_cd
        if grd_cd:    params["cond[grd_cd::EQ]"]   = grd_cd

        # 응답 컬럼 선택 (선택 사항)
        if selectable:
            params["selectable"] = selectable

        print(f"[호출 URL] https://apis.data.go.kr/{endpoint}")
        print(f"[파라미터] {json.dumps({k: v for k, v in params.items() if k != 'serviceKey'}, ensure_ascii=False, indent=2)}")

        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)

    def fetch_price_trend(self, exmn_ymd, se_cd=None, ctgry_cd=None, item_cd=None, vrty_cd=None, grd_cd=None, selectable=None, page=1, rows=10, return_type="json"):
        """
        가격 추이 정보를 조회합니다. (공식 엔드포인트: priceSequel/info)
        
        [필수]
          exmn_ymd  : 조사일자 (YYYYMMDD)
        """
        endpoint = "B552845/priceSequel/info"

        params = {
            "serviceKey": self.service_key,
            "pageNo":     str(page),
            "numOfRows":  str(rows),
            "returnType": return_type.upper(),
            "cond[exmn_ymd::EQ]": exmn_ymd,
        }

        if se_cd:     params["cond[se_cd::EQ]"]    = se_cd
        if ctgry_cd:  params["cond[ctgry_cd::EQ]"] = ctgry_cd
        if item_cd:   params["cond[item_cd::EQ]"]  = item_cd
        if vrty_cd:   params["cond[vrty_cd::EQ]"]  = vrty_cd
        if grd_cd:    params["cond[grd_cd::EQ]"]   = grd_cd

        if selectable:
            params["selectable"] = selectable

        print(f"[호출 URL] https://apis.data.go.kr/{endpoint}")
        print(f"[파라미터] {json.dumps({k: v for k, v in params.items() if k != 'serviceKey'}, ensure_ascii=False, indent=2)}")

        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)

    def fetch_price(self, endpoint_type, filters=None, page=1, rows=10, return_type="json"):
        """
        [하위 호환용] 범용 가격 정보 조회 메서드.
        새 코드에서는 fetch_region_price() 사용을 권장합니다.
        """
        endpoint = f"B552845/{endpoint_type}/price"
        params = {
            "serviceKey": self.service_key,
            "pageNo":     str(page),
            "numOfRows":  str(rows),
            "returnType": return_type.upper()
        }
        if filters:
            for key, val in filters.items():
                params[f"cond[{key}]"] = val

        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)


    def fetch_shipment_trend(
        self,
        # ── 필수 파라미터 ──────────────────────────────────────
        spmt_ymd,           # 출하일자 (YYYYMMDD), 예: '20250401' — EQ(일치) 조건
        # ── 선택 파라미터 (필터) ───────────────────────────────
        whsl_mrkt_cd=None,  # 도매시장코드
        corp_cd=None,       # 법인코드
        gds_lclsf_cd=None,  # 대분류코드
        gds_mclsf_cd=None,  # 중분류코드
        gds_sclsf_cd=None,  # 소분류코드
        # ── 응답 제어 파라미터 ─────────────────────────────────
        selectable=None,    # 원하는 응답 컬럼만 선택 (쉼표 구분)
        page=1,             # 페이지 번호 (기본값: 1)
        rows=10,            # 한 페이지 결과 수 (기본값: 10, 최대: 1000)
        return_type="json"  # 응답 형식: 'json' 또는 'xml'
    ):
        """
        출하일 추이 정보를 조회합니다.

        [필수]
          spmt_ymd  : 출하일자 (YYYYMMDD) — 정확히 일치하는 날짜만 검색

        [선택 필터]
          whsl_mrkt_cd  : 도매시장코드
          corp_cd       : 법인코드
          gds_lclsf_cd  : 대분류코드
          gds_mclsf_cd  : 중분류코드
          gds_sclsf_cd  : 소분류코드

        [응답 제어]
          selectable: 받고 싶은 컬럼만 콤마(,) 구분으로 지정
          page      : 페이지 번호
          rows      : 결과 수 (최대 1000)
          return_type: 'json' 또는 'xml'
        """
        endpoint = "B552845/shipmentSequel/info"

        # 기본 + 필수 파라미터
        params = {
            "serviceKey": self.service_key,
            "pageNo":     str(page),
            "numOfRows":  str(rows),
            "returnType": return_type.upper(),
            "cond[spmt_ymd::EQ]": spmt_ymd,
        }

        # 선택 필터 (값이 있을 때만 추가)
        if whsl_mrkt_cd: params["cond[whsl_mrkt_cd::EQ]"] = whsl_mrkt_cd
        if corp_cd:      params["cond[corp_cd::EQ]"]      = corp_cd
        if gds_lclsf_cd: params["cond[gds_lclsf_cd::EQ]"] = gds_lclsf_cd
        if gds_mclsf_cd: params["cond[gds_mclsf_cd::EQ]"] = gds_mclsf_cd
        if gds_sclsf_cd: params["cond[gds_sclsf_cd::EQ]"] = gds_sclsf_cd

        # 응답 컬럼 선택 (선택 사항)
        if selectable:
            params["selectable"] = selectable

        import json
        print(f"[호출 URL] https://apis.data.go.kr/{endpoint}")
        print(f"[파라미터] {json.dumps({k: v for k, v in params.items() if k != 'serviceKey'}, ensure_ascii=False, indent=2)}")

        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)

class NCSWholesaleClient(BaseAPIClient):
    """농림축산식품 공공데이터포털 (도매시장 경매정보) 전용 클라이언트"""
    def __init__(self, service_key):
        super().__init__(service_key, "http://211.237.50.150:7080/openapi")

    def fetch_auction_info(self, service_id, sale_date, whsal_cd, start_idx=1, rows=10, return_type="json"):
        """
        전국 도매시장 경매 정보를 가져오는 메서드
        """
        end_idx = start_idx + rows - 1
        # NCS API는 URL 경로에 파라미터가 포함되는 구조: {key}/{type}/{serviceId}/{start}/{end}
        endpoint = f"{self.service_key}/{return_type.lower()}/{service_id}/{start_idx}/{end_idx}"
        
        # 쿼리 스트링 파라미터
        params = {
            "SALEDATE": sale_date,
            "WHSALCD": whsal_cd
        }
        
        response = self._get(endpoint, params=params)
        if not response:
            return None

        if return_type.lower() == "json":
            try:
                return response.json()
            except:
                return response.text
        else:
            return self._prettify_xml(response.content)

# 편리한 사용을 위한 팩토리 함수 (선택 사항)
def get_at_client(key):
    return ATAPIClient(key)

def get_ncs_client(key):
    return NCSWholesaleClient(key)
