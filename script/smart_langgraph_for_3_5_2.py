

from __future__ import annotations
import json
import re
import os
import logging
from typing import Dict, Any, List, Optional, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# 로거 설정
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# 상수 설정
# =========================================================
YEAR_TO_FILENAME = {
    2020: "2020년_스마트폰_과의존_실태조_사보고서.pdf",
    2021: "2021년_스마트_과의존_실태조사_보고서.pdf",
    2022: "2022년_스마트폰_과의존_실태조사_보고서.pdf",
    2023: "2023년_스마트폰_과의존실태조사_최종보고서.pdf",
    2024: "2024_스마트폰_과의존_실태조사_본_보고서.pdf",
}
ALLOWED_FILES = list(YEAR_TO_FILENAME.values())


# 검색 파라미터 (기본값 / 재시도용)

# 기본 파라미터 (단일/2개 연도용)
DEFAULT_K_PER_QUERY = 10
DEFAULT_TOP_PARENTS = 30
DEFAULT_TOP_PARENTS_PER_FILE = 5

# 재시도용 파라미터
RETRY_K_PER_QUERY = 15
RETRY_TOP_PARENTS = 40
RETRY_TOP_PARENTS_PER_FILE = 8

# [신규] 연도별 검색 최소 보장 파라미터
MIN_DOCS_PER_YEAR = 6           # 연도당 최소 문서 수
MIN_PARENTS_PER_YEAR = 4        # 연도당 최소 parent 수
MIN_QUERIES_PER_YEAR = 2        # 연도당 최소 쿼리 수
MAX_DOCS_PER_YEAR_IN_CONTEXT = 5  # 컨텍스트 내 연도당 최대 문서 수

MAX_CHUNKS_PER_PARENT = 15
MAX_CHARS_PER_DOC = 20000
SUMMARY_TYPES = ['page_summary', 'table_summary']
MAX_RETRY_COUNT = 3

BOT_IDENTITY = """**2020~2024년 스마트폰 과의존 실태조사 보고서 분석 시스템**

**제공 가능한 정보:**
- 연도별 스마트폰 과의존 위험군 비율 및 추이
- 대상별(유아동, 청소년, 성인, 60대) 과의존 현황
- 학령별(초/중/고/대학생) 세부 분석
- 과의존 관련 요인 분석 (SNS, 숏폼, 게임 이용 등)
- 조사 방법론 및 표본 설계 정보
"""

# =========================================================
# LangGraph State 정의
# =========================================================
ValidationResult = Literal["PASS", "FAIL_NO_EVIDENCE", "FAIL_UNCLEAR", "FAIL_FORMAT"]

class GraphState(TypedDict):
    """LangGraph 상태 정의 - RAG 파이프라인 전체 상태를 관리"""
    
    # ====== 기본 입력 ========
    input: str  # 이번 턴 사용자 원문
    chat_history: List[BaseMessage]  # 멀티턴 대화 기록
    session_id: str  # 세션 식별자
    
    # ====== 라우팅 =========
    intent_raw: Optional[str]
    intent: Optional[str]
    is_chat_reference: Optional[bool]
    followup_type: Optional[str]
    
    # ==== 플래닝/리졸브 ====
    plan: Optional[Dict[str, Any]]
    resolved_question: Optional[str]
    previous_context: Optional[str]
    
    # ===== 쿼리 리라이트 ======
    rewritten_queries: Optional[List[str]]
    
    # ====== 검색 ======
    retrieval: Optional[Dict[str, Any]]
    context: Optional[str]
    extracted_figures: Optional[str]
    extracted_figures_json: Optional[Dict[str, Any]]
    compressed_context: Optional[str]
    
    # ===== 컨텍스트 정제 =====
    sanitized_context: Optional[str]
    
    # ===== 답변 생성 =====
    draft_answer: Optional[str]
    
    # ===== 다중 연도 =====
    year_extractions: Optional[List[Dict[str, Any]]]
    
    # ===== Safety/Validation =====
    safety_passed: Optional[bool]
    safety_issues: Optional[List[str]]
    validation_result: Optional[str]
    validation_reason: Optional[str]
    validator_output: Optional[Dict[str, Any]]
    
    # ===== 최종 포맷/출력 =====
    formatted_answer: Optional[str]
    final_answer: Optional[str]
    
    # ===== 리트라이/클래리파이 =====
    retry_count: Optional[int]
    retry_type: Optional[str]
    pending_clarification: Optional[str]
    clarification_context: Optional[Dict[str, Any]]
    
    # ===== 디버그/힌트 =====
    debug_info: Optional[Dict[str, Any]]
    dict_hint: Optional[Dict[str, Any]]
    used_default_years: Optional[bool]
    reranked_docs: Optional[List[Document]]

# =========================================================
# RAG Dictionary 로딩 및 인덱싱
# =========================================================
def load_rag_dict(rag_dict_path: str = 'rag_retrieval_dictionary.json') -> dict:
    """
    RAG Dictionary JSON 파일을 로드한다.
    
    Args:
        rag_dict_path: RAG Dictionary 파일 경로
        
    Returns:
        dict: 로드된 RAG Dictionary (실패 시 빈 딕셔너리)
    """
    try:
        if os.path.exists(rag_dict_path):
            with open(rag_dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"RAG Dictionary 로딩 실패: {e}")
    return {}

#
CROSS_ANALYSIS_PATTERNS = {
    "appendix_only": [
        (["유아동", "청소년", "성인", "60대"], ["과의존위험군", "일반사용자군"]),
        (["초등학생", "중학생", "고등학생","대학생"], ["과의존위험군", "일반사용자군"]),
        (["남성", "여성", "고등학생"], ["과의존위험군", "일반사용자군"]),
        (["대도시", "중소도시", "읍/면지역"], ["과의존위험군", "일반사용자군"])
        
    ],
    "total_only_in_main": ["고위험군", "잠재적위험군"],
}



def build_rag_dict_index(rag_dict: dict) -> dict:
    """
    RAG Dictionary에서 검색/환각방지에 사용할 인덱스를 구축한다.
    
    Args:
        rag_dict: 원본 RAG Dictionary
        
    Returns:
        dict: 인덱싱된 구조
            - core_synonyms: 핵심 용어의 동의어 목록
            - target_alias: target group 별칭 → 표준 그룹명 매핑
            - routing_patterns: (패턴, topic_code) 목록
            - disambig_rules: 혼동 방지 규칙
            - hallucination_rules: 할루시네이션 방지용 규칙
            - banner_structure: 통계표 배너 구조 정보
            - topic_taxonomy: 토픽 분류 체계
            - disambiguation_pairs: 혼동 가능한 쌍 + 구분 힌트
    """
    idx = {
        "core_synonyms": {},
        "target_alias": {},
        "routing_patterns": [],
        "disambig_rules": {},
        "hallucination_rules": {},
        "banner_structure": {},
        "topic_taxonomy": {},
        "disambiguation_pairs": [],
        "cross_analysis_rules": {},
        "banner_hierarchy": {},        
    }
    
    # ---- core_definitions: 핵심 용어 정의/동의어/비동의어 인덱싱 ----
    core_defs = (rag_dict or {}).get("core_definitions", {}) or {}
    for k, v in core_defs.items():
        if not isinstance(v, dict):
            continue
        syns = v.get("synonyms") or []
        if isinstance(syns, list) and syns:
            idx['core_synonyms'][str(k)] = [str(s).strip() for s in syns if str(s).strip()]
        not_syns = v.get("NOT_synonyms") or []
        if isinstance(not_syns, list) and not_syns:
            idx["disambig_rules"].setdefault(str(k), {})
            idx['disambig_rules'][str(k)]['not_synonyms'] = [str(s).strip() for s in not_syns if str(s).strip()]
    
    # ---- target_groups 그룹 별칭 처리 ----
    targets = (rag_dict or {}).get('target_groups', {}) or {}
    for tg_name, tg_obj in targets.items():
        if not isinstance(tg_obj, dict):
            continue
        also = tg_obj.get('also_called') or []
        if isinstance(also, list):
            for alias in also:
                a = str(alias).strip()
                if a:
                    idx['target_alias'][a] = str(tg_name)
    
    # ---- query_routing_guide: 쿼리 패턴 기반 topic_code 힌트 ----
    routing = ((rag_dict or {}).get("query_routing_guide", {}) or {}).get("patterns", []) or []
    for item in routing:
        if not isinstance(item, dict):
            continue
        pat = str(item.get("query_pattern", "") or "").strip()
        topic = str(item.get("primary_topic", "") or "").strip()
        if not pat or not topic:
            continue
        for p in [x.strip() for x in pat.split(",")]:
            if p:
                idx['routing_patterns'].append((p, topic))
    
    # ---- hallucination_prevention 규칙 인덱싱 ----
    h_rules = (rag_dict or {}).get('hallucination_prevention', {}) or {}
    for rule_key, rule_val in h_rules.items():
        if rule_key.startswith("_"):
            continue
        if isinstance(rule_val, dict):
            idx['hallucination_rules'][rule_key] = rule_val
    
    # ---- banner 구조 인덱싱 ----
    banner_info = (rag_dict or {}).get("stat_table_banner_structure", {}) or {}
    idx['banner_structure'] = banner_info
    
    # ---- topic_taxonomy: 토픽 분류 체계 ----
    topics = (rag_dict or {}).get('topic_taxonomy', {}) or {}
    for tk, tv in topics.items():
        if tk.startswith("_"):
            continue
        if isinstance(tv, dict):
            idx['topic_taxonomy'][tk] = tv
    
    # ---- disambiguation_pairs: 혼동 가능한 것들 ----
    disambig = (rag_dict or {}).get("disambiguation_rules", {}) or {}
    for dk, dv in disambig.items():
        if dk.startswith("_"):
            continue
        if isinstance(dv, dict):
            pair = dv.get('confusable_pair', [])
            distinction = dv.get("distinction", "")
            routing_hint = dv.get("routing_hint", "")
            if pair:
                idx["disambiguation_pairs"].append({
                    "rule_id": dk,
                    "pair": pair,
                    "distinction": distinction,
                    "routing_hint": routing_hint,
                })
                
    # [신규] banner 구조 상세 인덱싱
    banner_info = (rag_dict or {}).get("stat_table_banner_structure", {}) or {}
    idx['banner_structure'] = banner_info
    idx['banner_hierarchy'] = banner_info.get("banner_hierarchy", {})
    
    # [신규] hallucination_prevention에서 교차분석 규칙 추출
    h_rules = (rag_dict or {}).get('hallucination_prevention', {}) or {}
    idx['hallucination_rules'] = h_rules
    
    # 교차분석 금지 조합 추출
    h01 = h_rules.get("rule_H01_교차분석_부재", {})
    idx['cross_analysis_rules'] = {
        "prohibited_combinations": h01.get("prohibited_combinations", []),
        "exception_in_body": h01.get("exception_in_body", ""),
    }
    
    # 고위험군/잠재적위험군 세분화 범위
    h02 = h_rules.get("rule_H02_과의존수준_하위분류_범위", {})
    idx['high_risk_segmentation_rule'] = {
        "detail": h02.get("detail", ""),
        "prohibited_example": h02.get("prohibited_example", ""),
    }
    
    # 본문 vs 통계표 범위
    h06 = h_rules.get("rule_H06_본문_vs_통계표_데이터_범위", {})
    idx['body_vs_appendix'] = h06.get("implication_for_rag", {})
    
    return idx    return idx


# =========================================================
# [신규] RAG Dictionary 기반 키워드 추출 함수
# =========================================================
def extract_keywords_from_dict(
    text: str,
    rag_dict_index: dict,
    dict_hint: Optional[dict] = None
) -> List[str]:
    """
    RAG Dictionary 인덱스를 기반으로 텍스트에서 핵심 키워드를 추출한다.
    
    extract_core_keywords 함수를 대체하여 RAG Dictionary의 구조화된 정보를 활용.
    
    Args:
        text: 분석할 텍스트 (사용자 질문)
        rag_dict_index: build_rag_dict_index로 생성된 인덱스
        dict_hint: infer_dict_hint 결과 (있으면 우선 활용)
        
    Returns:
        추출된 핵심 키워드 목록 (중복 제거, 순서 유지)
    """
    keywords = []
    text_lower = (text or "").lower()
    
    # 1. dict_hint의 anchor_terms 우선 활용
    if dict_hint:
        anchor_terms = dict_hint.get("anchor_terms") or []
        keywords.extend([str(t).strip() for t in anchor_terms if str(t).strip()])
        
        # target_group도 키워드로 추가
        target_group = dict_hint.get("target_group", "")
        if target_group:
            keywords.append(target_group)
        
        # topic_code 관련 키워드 추가
        topic_code = dict_hint.get("topic_code", "")
        if topic_code:
            # topic_taxonomy에서 관련 키워드 추출
            topic_info = (rag_dict_index or {}).get("topic_taxonomy", {}).get(topic_code, {})
            if isinstance(topic_info, dict):
                topic_keywords = topic_info.get("keywords", [])
                if isinstance(topic_keywords, list):
                    keywords.extend([str(k).strip() for k in topic_keywords[:3] if str(k).strip()])
    
    # 2. routing_patterns에서 매칭되는 패턴의 키워드 추출
    routing_patterns = (rag_dict_index or {}).get("routing_patterns", [])
    for pattern, topic_code in routing_patterns:
        if pattern and pattern.lower() in text_lower:
            keywords.append(pattern)
    
    # 3. core_synonyms에서 텍스트에 포함된 핵심 용어 및 동의어 추출
    core_synonyms = (rag_dict_index or {}).get("core_synonyms", {})
    for term, synonyms in core_synonyms.items():
        # 핵심 용어가 텍스트에 있으면 추가
        if term and term.lower() in text_lower:
            keywords.append(term)
        # 동의어가 텍스트에 있으면 핵심 용어로 추가
        for syn in (synonyms or []):
            if syn and syn.lower() in text_lower:
                keywords.append(term)  # 동의어 대신 핵심 용어를 추가
                break
    
    # 4. target_alias에서 매칭되는 대상 그룹 추출
    target_alias = (rag_dict_index or {}).get("target_alias", {})
    for alias, canonical in target_alias.items():
        if alias and alias.lower() in text_lower:
            keywords.append(canonical)
    
    # 5. 텍스트에서 직접 대상 그룹 감지
    direct_targets = ['유아동', '청소년', '성인', '60대', '고령층', '시니어',
                      '초등학생', '중학생', '고등학생', '대학생']
    for target in direct_targets:
        if target in text:
            # 고령층/시니어는 60대로 통일
            if target in ['고령층', '시니어']:
                keywords.append('60대')
            else:
                keywords.append(target)
    
    # 6. 핵심 도메인 키워드 직접 감지 (RAG Dictionary에 없을 수 있는 것들)
    domain_keywords = {
        '과의존': ['과의존', '과의존률', '과의존위험군'],
        'SNS': ['SNS', '소셜네트워크', '소셜미디어'],
        '게임': ['게임', '온라인게임', '모바일게임'],
        '숏폼': ['숏폼', '쇼츠', '릴스', '틱톡'],
        '이용정도': ['이용수준', '이용빈도', '이용시간'],
        '이용률': ['이용률', '이용비율'],
        '예방교육': ['예방교육', '교육도움', '교육효과'],
    }
    
    for canonical, variants in domain_keywords.items():
        for variant in variants:
            if variant.lower() in text_lower:
                keywords.append(canonical)
                break
    
    # 중복 제거 및 순서 유지
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw = str(kw).strip()
        if kw and kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords

# =========================================================
# [신규] 연도 수에 따른 동적 파라미터 계산
# =========================================================
def calculate_search_params(num_years: int, retry_count: int = 0) -> Dict[str, int]:
    """
    연도 수에 따른 검색 파라미터를 동적으로 계산한다.
    
    멀티연도 검색 시 각 연도에서 충분한 문서를 확보하기 위해
    연도 수에 비례하여 파라미터를 조정.
    
    Args:
        num_years: 검색 대상 연도 수
        retry_count: 재시도 횟수 (재시도 시 파라미터 증가)
        
    Returns:
        dict: 검색 파라미터
            - k_per_query: 쿼리당 검색 문서 수
            - top_parents: 전체 선정할 parent 수
            - top_parents_per_file: 파일당 parent 수
            - max_queries: 최대 쿼리 수
    """
    # 기본 배수 (재시도 시 1.5배)
    base_multiplier = 1.5 if retry_count > 0 else 1.0
    
    # 연도 수에 비례하여 파라미터 조정
    # 기본값 + (연도당 추가분) * 배수
    k_per_query = int((DEFAULT_K_PER_QUERY + (num_years * 2)) * base_multiplier)
    top_parents = int((DEFAULT_TOP_PARENTS + (num_years * 4)) * base_multiplier)
    top_parents_per_file = int((DEFAULT_TOP_PARENTS_PER_FILE + num_years) * base_multiplier)
    max_queries = min(num_years * MIN_QUERIES_PER_YEAR + 4, 14)
    
    return {
        "k_per_query": min(k_per_query, 20),
        "top_parents": min(top_parents, 50),
        "top_parents_per_file": min(top_parents_per_file, 12),
        "max_queries": max_queries,
    }


# =========================================================
# [신규] 연도별 전용 쿼리 생성
# =========================================================
def generate_year_specific_queries(
    base_queries: List[str],
    year: int,
    resolved_question: str,
    rag_dict_index: dict,
    dict_hint: Optional[dict] = None
) -> List[str]:
    """
    특정 연도에 최적화된 검색 쿼리를 생성한다.
    
    RAG Dictionary 기반 키워드를 활용하여 각 연도에 특화된 쿼리 생성.
    
    Args:
        base_queries: 기본 쿼리 목록
        year: 대상 연도
        resolved_question: 해결된 질문
        rag_dict_index: RAG Dictionary 인덱스
        dict_hint: 힌트 딕셔너리
        
    Returns:
        해당 연도 전용 쿼리 목록 (최대 5개)
    """
    year_queries = []
    year_str = f"{year}년"
    
    # 1. 기존 쿼리에서 연도 교체
    for q in base_queries:
        # 기존 연도 패턴 제거
        clean_q = re.sub(r'20[2][0-4]년?\s*', '', q).strip()
        clean_q = re.sub(r'20[2][0-4]~?20[2][0-4]년?\s*', '', clean_q).strip()
        clean_q = re.sub(r'20[2][0-4][-~]20[2][0-4]년?\s*', '', clean_q).strip()
        
        if clean_q:
            year_query = f"{year_str} {clean_q}"
            if year_query not in year_queries:
                year_queries.append(year_query)
    
    # 2. RAG Dictionary 기반 키워드 + 연도 조합
    keywords = extract_keywords_from_dict(resolved_question, rag_dict_index, dict_hint)
    for kw in keywords[:3]:
        kw_query = f"{year_str} {kw}"
        if kw_query not in year_queries:
            year_queries.append(kw_query)
    
    # 3. anchor_terms 활용
    if dict_hint:
        anchors = dict_hint.get("anchor_terms", [])
        for anchor in anchors[:2]:
            anchor_query = f"{year_str} {anchor}"
            if anchor_query not in year_queries:
                year_queries.append(anchor_query)
    
    # 4. 파일명 포함 쿼리 (해당 연도 보고서 명시)
    fn = YEAR_TO_FILENAME.get(year, "")
    if fn:
        # 파일명에서 핵심 부분 추출
        fn_query = f"{resolved_question[:50]} {year}년 스마트폰 과의존 실태조사"
        if fn_query not in year_queries:
            year_queries.append(fn_query)
    
    return year_queries[:5]


# =========================================================
# [신규] 문서에서 연도 추출
# =========================================================
def extract_year_from_doc(doc: Document) -> Optional[int]:
    """
    문서 메타데이터에서 연도를 추출한다.
    
    Args:
        doc: 문서 객체
        
    Returns:
        추출된 연도 (없으면 None)
    """
    fn = doc.metadata.get("file_name", "")
    
    # 파일명에서 연도 추출
    match = re.search(r'(20[2][0-4])', fn)
    if match:
        return int(match.group(1))
    
    # 메타데이터에 직접 연도가 있는 경우
    if "_year" in doc.metadata:
        try:
            return int(doc.metadata["_year"])
        except (ValueError, TypeError):
            pass
    
    return None

# =========================================================
# RAG Dictionary 힌트 추론 함수
# =========================================================
def infer_dict_hint(text: str, context_text: str = "", rag_dict_index: dict = None) -> dict:
    """
    사용자 질문(text)과 이전대화(context_text)를 바탕으로 힌트를 추론한다.
    
    Args:
        text: 사용자 질문
        context_text: 이전 대화 컨텍스트
        rag_dict_index: RAG Dictionary 인덱스
        
    Returns:
        dict: 추론된 힌트
    """
    if rag_dict_index is None:
        rag_dict_index = {}
    
    q = (text or "").strip()
    q_low = q.lower()
    
    # ----- 1) target group 감지 -----
    target_group = ""
    for t in ['유아동', '청소년', '성인', '60대', '고령층', '시니어']:
        if t in q:
            target_group = "60대" if t in ['60대', '고령층', '시니어'] else t
            break
    
    if not target_group:
        for alias, canon in (rag_dict_index.get("target_alias") or {}).items():
            if alias and alias in q:
                target_group = canon
                break
    
    # ----- 2) 토픽 코드 감지 -----
    topic_hits = {}
    for pat, tcode in (rag_dict_index.get("routing_patterns") or []):
        if pat and pat.lower() in q_low:
            topic_hits[tcode] = topic_hits.get(tcode, 0) + 1
    
    topic_code = sorted(topic_hits.items(), key=lambda x: (-x[1], x[0]))[0][0] if topic_hits else ""
    is_rag_like = bool(topic_code)
    
    anchor_terms, avoid_terms = [], []
    
    # ----- 3) 대표 혼동쌍 앵커/회피 토큰 부여 -----
    if "이용률" in q or "몇 %" in q or "비율" in q:
        anchor_terms.append("이용률")
        avoid_terms.append("이용정도")
    if "이용정도" in q or "이용 빈도" in q or "빈도" in q:
        anchor_terms.append("이용정도")
        avoid_terms.append("이용률")
    
    if "과다이용" in q or "많이 쓴다고" in q or "과하게" in q:
        anchor_terms.append("과다이용 인식")
        avoid_terms.append("과의존위험군 비율")
    if ("위험군" in q) or ("고위험" in q) or ("잠재적" in q) or ("일반사용자군" in q):
        anchor_terms.extend(["과의존위험군", "비율"])
        avoid_terms.append("과다이용 인식")
    
    # ----- 4) 숏폼/플랫폼 키워드면 토픽 강제 -----
    if any(x in q for x in ['숏폼', '쇼츠', '릴스', '틱톡']):
        anchor_terms.extend(['숏폼', "플랫폼"])
        if not topic_code:
            topic_code = "T06"
            is_rag_like = True
    
    # ----- 5) 핵심 용어(과의존) 동의어를 앵커로 추가 -----
    if "과의존" in q or "중독" in q or "과몰입" in q or "과사용" in q:
        syns = (rag_dict_index.get("core_synonyms") or {}).get("과의존") or []
        for s in syns[:2]:
            if s not in anchor_terms:
                anchor_terms.append(s)
    
    # ----- 6) SNS, 게임 관련 키워드 앵커 추가 -----
    if "SNS" in q or "소셜" in q:
        anchor_terms.append("SNS")
    if "게임" in q:
        anchor_terms.append("게임")
    if "예방교육" in q or "교육" in q:
        anchor_terms.append("예방교육")
    
    def _uniq(lst):
        seen, out = set(), []
        for x in lst:
            x = str(x).strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    # ====== 본문 vs 부록 범위 감지 ======
    needs_appendix_table = False
    scope_warnings = []
    
    _target_kws = ["유아동", "청소년", "성인", "60대", "고령층", "시니어",
                   "초등학생", "중학생", "고등학생", "대학생"]
    _overdep_compare_kws = ["과의존위험군", "과의존 위험군", "일반사용자군", "일반 사용자군",
                            "과의존여부", "과의존 여부", "위험군별", "과의존수준별",
                            "과의존군", "일반군"]
    has_target_kw = any(t in q for t in _target_kws)
    has_overdep_compare = any(k in q for k in _overdep_compare_kws)
    
    _rate_only_markers = ["과의존률", "과의존율", "비율", "%", "퍼센트", "구분", "분류", "표로", "정리", "추이"]
    _other_metric_markers = ["이용시간", "이용정도", "이용빈도", "이용행태", "요인", "영향", "상관", "비교", "차이", "분석"]
    
    is_overdep_rate_only = (
        any(k in q for k in _rate_only_markers) and
        not any(k in q for k in _other_metric_markers)
    )
    
    if has_target_kw and has_overdep_compare and (not is_overdep_rate_only):
        needs_appendix_table = True
        scope_warnings.append(
            "★ 이 질문은 특정 연령대 내에서 과의존 여부별(과의존위험군 vs 일반사용자군) 비교를 요청합니다."
        )
    
    _high_risk_kws = ['고위험군', "잠재적위험군", '잠재적 위험군', '고 위험군']
    has_high_risk = any(k in q for k in _high_risk_kws)
    if has_target_kw and has_high_risk:
        scope_warnings.append(
            "★ 고위험군/잠재적위험군 세분화는 전체(B2) 기준에서만 존재합니다."
        )
    
    return {
        "is_rag_like": is_rag_like,
        "topic_code": topic_code,
        'target_group': target_group,
        "anchor_terms": _uniq(anchor_terms),
        "avoid_terms": _uniq(avoid_terms),
        "needs_appendix_table": needs_appendix_table,
        "scope_warnings": scope_warnings
    }


def augment_queries_with_anchors(queries: list, anchor_terms: list, max_extra: int = 2) -> list:
    """
    이미 만들어진 쿼리 리스트에 anchor_terms를 덧붙여 추가 쿼리를 생성한다.
    """
    if not isinstance(queries, list):
        return queries
    
    anchors = [a for a in (anchor_terms or []) if isinstance(a, str) and a.strip()]
    if not anchors:
        return queries
    
    out = []
    for q in queries:
        s = str(q).strip()
        if s and s not in out:
            out.append(s)
    
    extra_added = 0
    for q in list(out):
        if extra_added >= max_extra:
            break
        add = " ".join(anchors[:2])
        cand = f"{q} {add}".strip()
        if len(cand) >= 6 and cand not in out:
            out.append(cand)
            extra_added += 1
    return out



def build_context_guard(
    dict_hint: dict, 
    resolved_question: str = "", 
    similar_info: dict = None,
    rag_dict_index: dict = None  # [신규]
) -> str:
    lines = []
    
    scope_warnings = (dict_hint or {}).get("scope_warnings") or []
    for w in scope_warnings:
        lines.append(w)
    
    if (dict_hint or {}).get("needs_appendix_table"):
        lines.append(
            "★ 부록 통계표 데이터가 필요한 질문입니다. "
            "본문(제3장)의 전체 기준 수치를 특정 연령대/학령/성별의 "
            "과의존여부별 수치로 오인하여 응답하지 마십시오."
        )
    
    # [신규] 유사 내용 활용 안내
    if similar_info and similar_info.get("has_similar"):
        missing = similar_info.get("missing_concept", "")
        similar = similar_info.get("similar_concept", "")
        explanation = similar_info.get("explanation", "")
        lines.append(
            f"★ [유사 데이터 활용] 요청하신 '{missing}' 데이터는 본 조사에 포함되어 있지 않습니다. "
            f"대신 '{similar}' 데이터가 있으므로 이를 활용하여 답변하십시오. ({explanation})"
            )
        
    lines.append(
        "★ [공통] 컨텍스트에 명시적으로 존재하는 수치와 출처만 인용하십시오."
    )
    
    lines.append(
        "★ [본문 vs 부록 구분] 본문 통계표의 '전체' 행의 과의존위험군/일반사용자군 수치는 "
        "전체 인구 기준입니다. 특정 연령대 내 과의존여부별 비교가 필요하면, "
        "해당 연령대가 명시된 부록 통계표에서 수치를 확인하십시오."
    )
    
    return "\n".join(lines) if lines else ""

def detect_cross_analysis_need(
    query: str,
    context: str,
    dict_hint: dict,
    rag_dict_index: dict  # [신규] 파라미터 추가
) -> Dict[str, Any]:
    """JSON 기반으로 교차분석 필요 여부 감지"""
    
    result = {
        "needs_appendix": False,
        "cross_type": None,
        "target_group": None,
        "cross_condition": None,
        "reason": None,
        "retrieval_hint": None,  # [신규]
    }
    
    # [신규] JSON에서 배너 구조 로드
    banner_hierarchy = rag_dict_index.get("banner_hierarchy", {})
    cross_rules = rag_dict_index.get("cross_analysis_rules", {})
    high_risk_rule = rag_dict_index.get("high_risk_segmentation_rule", {})
    
    # B3_연령대별에서 카테고리 추출
    b3 = banner_hierarchy.get("B3_연령대별", {})
    target_groups = b3.get("categories", ["유아동", "청소년", "성인", "60대"])
    target_sub_rows = b3.get("per_category_sub_rows", ["전체", "과의존위험군", "일반사용자군"])
    
    # B5_학령별에서 카테고리 추출
    b5 = banner_hierarchy.get("B5_학령별", {})
    grade_groups = b5.get("categories", ["초등학생", "중학생", "고등학생", "대학생"])
    
    # B2_과의존수준별에서 세분화 가능 항목 추출
    b2 = banner_hierarchy.get("B2_과의존수준별", {})
    b2_sub_rows = b2.get("sub_rows", ["과의존위험군", "고위험군", "잠재적위험군", "일반사용자군"])
    
    # [신규] 고위험군/잠재적위험군은 B2(전체)에서만 존재
    high_risk_only_in_b2 = ["고위험군", "잠재적위험군"]
    
    query_lower = query.lower()
    
    # 1. 연령대/학령 + 고위험군/잠재적위험군 조합 요청 감지
    found_target = next((tg for tg in target_groups if tg in query), None)
    found_grade = next((gg for gg in grade_groups if gg in query), None)
    found_high_risk = any(hr in query for hr in high_risk_only_in_b2)
    
    if (found_target or found_grade) and found_high_risk:
        target = found_target or found_grade
        result["cross_type"] = "high_risk_by_target"
        result["target_group"] = target
        result["cross_condition"] = "고위험군/잠재적위험군"
        result["needs_appendix"] = False  # 부록에도 없음
        result["reason"] = high_risk_rule.get("detail", 
            f"고위험군/잠재적위험군 세분화는 전체(B2) 기준에서만 존재. '{target}' 내 세분화 없음")
        result["retrieval_hint"] = "DATA_NOT_EXIST"  # [신규] 데이터 자체가 없음 표시
        return result
    
    # 2. 배너 간 교차 요청 감지 (성별×연령대 등)
    prohibited = cross_rules.get("prohibited_combinations", [])
    # JSON에서 금지 조합 패턴 파싱하여 매칭
    for combo in prohibited:
        # 예: "성별 × 연령대별 (예: '남성 청소년의 과의존위험군 비율')"
        if "성별" in combo and "연령대" in combo:
            gender_found = any(g in query for g in ["남성", "여성", "남자", "여자"])
            target_found = any(t in query for t in target_groups)
            if gender_found and target_found:
                result["cross_type"] = "gender_target_cross"
                result["needs_appendix"] = False
                result["reason"] = "성별×연령대별 교차 데이터는 통계표에 존재하지 않음"
                result["retrieval_hint"] = "DATA_NOT_EXIST"
                return result
    
    # 3. 연령대/학령 × 과의존여부별 교차 (부록에 존재)
    overdep_conditions = ["과의존위험군", "일반사용자군", "과의존여부", "위험군별"]
    found_overdep = any(od in query for od in overdep_conditions)
    
    if (found_target or found_grade) and found_overdep:
        target = found_target or found_grade
        result["cross_type"] = "target_overdep"
        result["target_group"] = target
        result["cross_condition"] = "과의존여부별"
        
        # 컨텍스트에 해당 교차 데이터가 있는지 확인
        cross_pattern = rf"{target}.*?(과의존위험군|일반사용자군).*?\d"
        if not re.search(cross_pattern, context, re.IGNORECASE):
            result["needs_appendix"] = True
            result["reason"] = f"'{target}' 내 과의존여부별 데이터가 본문에서 확인되지 않음. 부록 통계표 검색 필요"
            result["retrieval_hint"] = "SEARCH_APPENDIX"
    
    return result


def detect_scope_mismatch(answer: str, context: str, dict_hint: dict) -> List[str]:
    """
    답변이 '잘못된 범위'의 데이터를 사용했는지 감지한다.
    """
    issues = []
    
    if not (dict_hint or {}).get("needs_appendix_table"):
        return issues
    
    target_group = (dict_hint or {}).get("target_group", "")
    if not target_group:
        return issues
    
    _overdep_kws = ['과의존위험군', '일반사용자군', '일반군', '과의존군']
    has_target_in_answer = target_group in answer
    has_overdep_in_answer = any(k in answer for k in _overdep_kws)
    
    if has_target_in_answer and has_overdep_in_answer:
        blocks = context.split("---")
        found_matching_block = False
        
        for block in blocks:
            block_text = block.strip()
            if not block_text:
                continue
            
            has_target_in_block = target_group.lower() in block_text.lower()
            has_overdep_in_block = any(k in block_text for k in _overdep_kws)
            
            if has_target_in_block and has_overdep_in_block:
                found_matching_block = True
                break
        
        if not found_matching_block:
            issues.append(
                f"⚠ '{target_group}'의 과의존 여부별 비교 데이터가 컨텍스트에서 확인되지 않았습니다."
            )
    
    return issues


#  누락 연도 타겟 재검색
# =========================================================
def targeted_year_search(
    target_years: List[int],
    query: str,
    vectorstore,
    rag_dict_index: dict,
    dict_hint: dict,
    k_per_query: int = 10
) -> str:
    """
    특정 연도에 대해 타겟 검색을 수행한다.
    
    누락된 연도가 감지되었을 때 해당 연도에 집중하여 추가 검색.
    
    Args:
        target_years: 검색 대상 연도 목록
        query: 검색 쿼리
        vectorstore: 벡터스토어
        rag_dict_index: RAG Dictionary 인덱스
        dict_hint: RAG Dictionary 힌트
        k_per_query: 쿼리당 검색 수
        
    Returns:
        검색된 컨텍스트 문자열
    """
    blocks = []
    
    for year in target_years:
        fn = YEAR_TO_FILENAME.get(year)
        if not fn:
            continue
        
        file_filter = {'$and': [
            {'doc_type': {"$in": SUMMARY_TYPES}},
            {'file_name': fn}
        ]}
        
        # RAG Dictionary 기반 키워드 활용
        keywords = extract_keywords_from_dict(query, rag_dict_index, dict_hint)
        keyword_str = " ".join(keywords[:3])
        
        year_query = f"{year}년 {query} {keyword_str}".strip()
        
        try:
            hits = vectorstore.similarity_search_with_relevance_scores(
                year_query, k=k_per_query, filter=file_filter
            )
            
            for doc, score in hits[:5]:
                m = doc.metadata
                text = doc.page_content[:5000]
                blocks.append(
                    f"[{year}년 보완검색] {m.get('file_name', '')} "
                    f"(p.{m.get('page', '?')}, score={score:.3f})\n{text}"
                )
        except Exception:
            pass
    
    return "\n\n---\n\n".join(blocks)
def create_node_functions(vectorstore, llms, status_callback, rag_dict_index):
    """
    모든 노드 함수를 생성하는 팩토리 함수.
    
    Args:
        vectorstore: ChromaDB 벡터스토어
        llms: LLM 딕셔너리
        status_callback: 상태 업데이트 콜백 함수
        rag_dict_index: RAG Dictionary 인덱스
        
    Returns:
        dict: 노드 이름 → 노드 함수 매핑
    """
    
    # LLM 참조
    router_llm = llms["router"]
    chat_refer_llm = llms["chat_refer"]
    parse_year_llm = llms["parse_year"]
    followup_llm = llms["followup"]
    casual_llm = llms["casual"]
    main_llm = llms["main"]
    rewrite_llm = llms["rewrite"]
    validator_llm = llms["validator"]
    
    # =========================================================
    # 프롬프트 정의 (원본과 동일)
    # =========================================================
    
    chat_refer_prompt = """
[역할]
당신은 멀티턴 대화에서 '현재 질문(curr)이 이전 대화(context)에 의존하는 후속질문인지'만 판정하는 이진 분류기입니다.

<CONTEXT>
{context}
</CONTEXT>

[출력 제약]
- 오직 True 또는 False 한 단어만 출력하세요.

[True(후속질문) 조건]
- curr가 prev의 결과/수치/비교/해석을 전제로 추가 질문하는 경우
- curr가 prev의 핵심 대상/지표를 유지하면서 조건만 바꾸는 경우
- curr가 생략/대명사 중심이라 context 없이는 불명확한 경우

[False(비후속) 조건]
- curr가 context 없이도 독립 질문으로 완결되는 경우
- 완전히 새로운 주제로서 prev를 전제로 하지 않는 경우
""".strip()

    parse_year_prompt_text = """
[ROLE]
당신은 입력 텍스트에서 연도/연도범위를 추출하여, 사용 가능한 연도만 반환하는 파서입니다.

<BASE_YEAR>
{base_year}
</BASE_YEAR>
<AVAILABLE_YEARS>
{available_years}
</AVAILABLE_YEARS>

[OUTPUT FORMAT]
- 출력은 반드시 아래 형식만 허용:
  {{"years":[2022,2023,2024]}}
  {{"years":[]}}
- 공백, 개행, 설명 금지
- years 배열은 오름차순 정렬

[ALGORITHM]
1) 연도 범위 표현(2020~2024, 2020년부터 2024년까지 등)을 먼저 처리
2) 단일 연도 표현 추출
3) AVAILABLE_YEARS에 없는 연도 제거
4) 오름차순 정렬 후 JSON 출력
"""

    followup_rewrite_prompt = """
[역할]
당신은 후속질문(curr)을 검색에 적합한 '단독 질문'으로 재작성하는 모듈입니다.

[입력]
context: {context}
curr: {curr}

[목표]
- curr가 context에 의존하는 생략/대명사를 복원
- context 없이도 의미가 완결되는 질문 1개로 재작성

[규칙]
- 질문 한 문장만 출력
- context에 없는 정보를 만들어내지 않음
- 기간이 context에 있고 curr가 기간을 말하지 않으면 유지
""".strip()

    personal_memory_prompt = """
[역할]
사용자 질문이 '사용자 본인에 대한 정보'를 기억하는지 확인하는 질문인지 판정합니다.

[True 예시]
- "제가 누구라고요?", "제 이름 기억하나요?", "아까 내가 뭐라고 말했지?"

[False 예시]
- 보고서 수치/분석 요청, 시스템 질문, 일반 조언

[출력]
True 또는 False만 출력
""".strip()

    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "사용자 질문을 분류하는 라우터입니다.\n"
         "이 시스템은 '스마트폰 과의존 실태조사 보고서(2020~2024)' 전문 RAG를 포함합니다.\n\n"
         "[LABELS]\n"
         "SMALLTALK: 인사/감사/잡담/일상대화\n"
         "META: 시스템/모델/데이터 범위 질문\n"
         "RAG: 보고서 내용 기반 질문\n"
         "GENERAL_ADVICE: 과의존 줄이는 방법 등 일반 조언\n\n"
         "[OUTPUT RULE]\n"
         "라벨명 1개만 출력: SMALLTALK 또는 META 또는 RAG 또는 GENERAL_ADVICE\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    smalltalk_system = """
[ROLE]
'스마트폰 과의존 실태조사 보고서 분석 시스템'의 스몰토크 응답기입니다.
한국어 존댓말, 1~2문장으로 짧게.
보고서/수치/출처 언급 금지.
""".strip()

    meta_system = """
[ROLE]
시스템/사용법/데이터 범위를 설명하는 안내자입니다.
통계 수치를 만들어내지 마십시오.
한국어 존댓말, 3~6줄로 간단명료.
""".strip()

    general_advice_system = """
[ROLE]
스마트폰 과의존 관련 일반적인 생활/행동 가이드를 제공합니다.
통계 수치를 만들어내지 마십시오.
실행 팁 5~7개, 주의사항 1~2줄.
""".strip()

    planner_prompt_text = (
        "스마트폰 과의존 실태조사 보고서(2020~2024년) 검색 계획 수립기입니다.\n"
        "반드시 유효한 JSON만 출력하세요.\n\n"
        "[연도/파일 규칙]\n"
        "- 연도를 임의로 확장하지 않음\n"
        "- 별도 연도 지정이 없으면 기본 연도(2023, 2024) 적용\n\n"
        "허용 파일명:\n" +
        "\n".join([f"- {y}년: {fn}" for y, fn in YEAR_TO_FILENAME.items()]) +
        "\n\n[queries 생성 규칙]\n"
        "queries는 최소 3개, 연도 수에 따라 증가\n"
        "모든 연도가 최소 1개 쿼리에 포함되어야 함\n\n"
        "JSON 스키마:\n"
        "{{\n"
        '  "resolved_question": "완전한 질문",\n'
        '  "years": [2020, ...],\n'
        '  "file_name_filters": ["파일명"],\n'
        '  "queries": ["쿼리1", "쿼리2", ...]\n'
        "}}"
    )

    _rewrite_prompt_25 = ChatPromptTemplate.from_messages([
        ("system",
         "검색 쿼리 최적화 전문가입니다.\n"
         "불필요한 조사/어미 제거, 핵심 키워드 추출, 동의어 확장.\n"
         "JSON: {{\"optimized_queries\": [\"쿼리1\", \"쿼리2\", ...]}}"
        ),
        ("human",
         "원본 질문: {resolved_question}\n원본 쿼리: {queries}\n연도: {years}\n\nJSON:")
    ])

    _answer_prompt_25 = ChatPromptTemplate.from_messages([
    ("system",
     "스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n\n"
     "원칙:\n"
     "1. CONTEXT에서 수치 인용 필수\n"
     "2. 모든 연도별 수치에 출처 표기: (파일명.pdf p.00)\n"
     "3. 변화량(%p) 명시(가능할 때)\n"
     "4. CONTEXT에 없으면 '검색 결과에 포함되지 않았습니다' 명시\n\n"
     "{context_guard}"
    ),
    ("human",
     "[질문]\n{input}\n\n[CONTEXT]\n{context}\n\n답변:")
    ])

    _answer_retry_prompt_25 = ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n\n"
         "⚠️ 이전 문제: {previous_issue}\n\n"
         "수정 지침:\n"
         "1. 모든 수치에 출처 형식: (파일명.pdf p.00)\n"
         "2. CONTEXT에서 직접 인용만\n"
         "3. 없는 정보는 '포함되지 않았습니다' 명시\n\n"
         "{context_guard}"
        ),
        ("human",
         "[질문]\n{input}\n\n[CONTEXT]\n{context}\n\n수정된 답변:")
    ])

    _validator_prompt_25 = ChatPromptTemplate.from_messages([
        ("system",
         "답변 품질 검수기입니다.\n\n"
         "분류:\n"
         "- PASS: 양호\n"
         "- FAIL_NO_EVIDENCE: 근거 부족\n"
         "- FAIL_UNCLEAR: 질문 불명확\n"
         "- FAIL_FORMAT: 형식 문제\n\n"
         "{context_guard}\n\n"
         "JSON: {{\"result\": \"PASS|FAIL_...\", \"reason\": \"...\", "
         "\"clarify_question\": \"...\", \"corrected_answer\": \"...\"}}"
        ),
        ("human",
         "[질문]\n{input}\n\n[CONTEXT]\n{context}\n\n[답변]\n{answer}\n\nJSON:")
    ])

    EXTRACT_FIGURES_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 통계 보고서에서 핵심 수치만 정확히 발췌하는 추출기입니다.\n\n"
         "[절대 규칙]\n"
         "1. 컨텍스트에 명시된 수치만 발췌. 추론·보간·반올림 금지.\n"
         "2. 해당 수치가 없으면 'N/A'로 표기.\n"
         "3. JSON만 출력.\n\n"
         "[출력 형식]\n"
         "{{\n"
         '  "연도별_수치": [\n'
         '    {{"연도": 2020, "전체": "XX.X%", "유아동": "XX.X%", "청소년": "XX.X%", "성인": "XX.X%", "60대": "XX.X%"}},\n'
         "  ]\n"
         "}}"
        ),
        ("human",
         "[추출 대상 질문]\n{resolved_question}\n\n"
         "[컨텍스트]\n{context}\n\n"
         "JSON:")
    ])

    # =========================================================
    # 헬퍼 함수들
    # =========================================================
    _True_False_re = re.compile(r'\b(True|False)\b')

    def is_chat_reference_question(context: str, curr: str) -> bool:
        """LLM 기반: 현재 질문이 이전 대화 맥락의 후속질문인지 판정."""
        system_chat_refer_prompt = ChatPromptTemplate.from_messages([
            ("system", chat_refer_prompt),
            ("human", '{curr}')
        ])
        chat_refer_chain = system_chat_refer_prompt | chat_refer_llm | StrOutputParser()
        result = chat_refer_chain.invoke({'context': context, "curr": curr})
        result = result.strip()
        result_re = _True_False_re.search(result)
        if not result_re:
            return False
        return result_re.group(1) == 'True'

    def parse_year_range(user_input: str) -> List[int]:
        """사용자 입력 텍스트에서 연도/연도범위를 추출."""
        available_years = [2020, 2021, 2022, 2023, 2024]
        system_parse_year_prompt = ChatPromptTemplate.from_messages([
            ("system", parse_year_prompt_text),
            ("human", '{text}')
        ])
        parse_year_chain = system_parse_year_prompt | parse_year_llm | StrOutputParser()
        result_years = parse_year_chain.invoke({
            'text': user_input,
            'available_years': available_years,
            'base_year': "2026"
        })
        try:
            obj = json.loads(result_years.strip())
            years = obj.get("years", [])
            years = [int(y) for y in years if str(y).isdigit()]
            years = sorted([y for y in years if y in available_years])
        except Exception:
            years = []
        return years

    def classify_followup_type(user_input: str, context: str) -> str:
        """후속질문을 standalone 질문으로 재작성."""
        system_followup_rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", followup_rewrite_prompt),
            ("human", '{curr}')
        ])
        followup_answer_chain = system_followup_rewrite_prompt | followup_llm | StrOutputParser()
        follow_result = followup_answer_chain.invoke({'context': context, "curr": user_input})
        follow_question = follow_result.strip()
        return follow_question

    def is_personal_memory_question(context: str, curr: str) -> bool:
        """사용자 개인 정보 기억 관련 질문인지 판정."""
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", personal_memory_prompt),
            ("human", "{curr}")
        ])
        chain = system_prompt | router_llm | StrOutputParser()
        result = (chain.invoke({"context": context, "curr": curr}) or "").strip()
        m = _True_False_re.search(result)
        if not m:
            return False
        return m.group(1) == "True"

    def _norm_label(x: str) -> str:
        if x is None:
            return ""
        return str(x).strip().upper()

    def _keyword_boost_score(doc: Document, query: str, dict_hint: Optional[dict] = None) -> float:
        """키워드 기반 부스트 점수 계산."""
        text = (doc.page_content or "").lower()
        query_terms = re.findall(r"[가-힣a-zA-Z0-9]+", (query or "").lower())
        boost = 0.0

        for term in query_terms:
            if len(term) >= 2 and term in text:
                boost += 0.02

        if isinstance(dict_hint, dict):
            anchor_terms = dict_hint.get("anchor_terms") or []
            avoid_terms = dict_hint.get("avoid_terms") or []

            for a in anchor_terms:
                a = str(a).strip().lower()
                if len(a) >= 2 and a in text:
                    boost += 0.03

            for v in avoid_terms:
                v = str(v).strip().lower()
                if len(v) >= 2 and v in text:
                    boost -= 0.015

        boost = max(-0.05, min(boost, 0.20))

        if isinstance(dict_hint, dict) and dict_hint.get("needs_appendix_table"):
            target_group = dict_hint.get("target_group", "")
            if target_group and target_group.lower() in text:
                _overdep_markers = ["과의존위험군", "일반사용자군"]
                if all(m in text for m in _overdep_markers):
                    boost += 0.05
            if "전체" in text and target_group and target_group.lower() not in text:
                if any(m in text for m in ["과의존위험군", "일반사용자군"]):
                    boost -= 0.02

        boost = max(-0.08, min(boost, 0.25))
        return boost

    def _extract_years_from_chat_history(chat_history: List[BaseMessage], available_years: List[int] = None) -> List[int]:
        """chat_history에서 연도 패턴을 추출."""
        if available_years is None:
            available_years = [2020, 2021, 2022, 2023, 2024]

        year_pattern = re.compile(r'(20[12]\d)')
        found_years = set()
        for msg in chat_history:
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            matches = year_pattern.findall(content)
            for m in matches:
                y = int(m)
                if y in available_years:
                    found_years.add(y)

        return sorted(found_years)

    def _extract_last_context_hints(chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """chat_history에서 직전 대화의 핵심 힌트를 추출."""
        years = _extract_years_from_chat_history(chat_history)
        last_user_query = ""
        last_ai_snippet = ""
        
        for msg in reversed(chat_history):
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", "") or ""
            if msg_type == "ai" and not last_ai_snippet:
                last_ai_snippet = content[:200]
            elif msg_type == "human" and not last_user_query:
                last_user_query = content
            if last_user_query and last_ai_snippet:
                break

        return {
            "years": years,
            "last_user_query": last_user_query,
            "last_ai_answer_snippet": last_ai_snippet,
        }

    def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
        """LLM 출력에서 JSON을 안전하게 파싱."""
        if not text:
            return {}
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)

        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            truncated = text
            last_complete = truncated.rfind("}")
            if last_complete > 0:
                truncated = truncated[:last_complete + 1]
            return json.loads(truncated)
        except Exception:
            pass

        return {}

    def _append_year_confirmation(answer: str, state: dict) -> str:
        """연도 미지정 시 기본 연도 사용 안내 메시지를 추가."""
        years = state.get("plan", {}).get("years", [2023, 2024])
        year_str = ", ".join([f"{y}년" for y in years])
        confirmation_msg = (
            f"\n\n---\n"
            f"📌 **연도 확인 요청**: 질문에 특정 연도가 명시되지 않아 "
            f"**최근 데이터({year_str})**를 기준으로 답변드렸습니다. "
            f"다른 연도(2020~2024년)의 정보가 필요하시면 말씀해 주세요."
        )
        return answer + confirmation_msg

    def _reset_turn_fields(state: GraphState) -> None:
        """이번 턴의 필드를 초기화."""
        for k in [
            "intent_raw", "intent", "is_chat_reference", "followup_type",
            "plan", "resolved_question", "previous_context",
            "rewritten_queries", "retrieval", "context", "reranked_docs",
            "compressed_context", "sanitized_context",
            "draft_answer", "final_answer",
            "validation_result", "validation_reason", "validator_output",
            "extracted_figures", "extracted_figures_json", "year_extractions",
            "pending_clarification",
            "used_default_years",
        ]:
            state[k] = None

    # =========================================================
    # 노드 함수들
    # =========================================================
    
    def route_intent(state: GraphState) -> GraphState:
        """라우팅 노드 — 의도 분류 + 맥락 처리."""
        status_callback("🔄 질문 분석 중...")
        
        try:
            user_input = state.get("input", "")
            chat_history = state.get("chat_history", [])
            session_id = state.get("session_id", None)
            clarification_ctx = state.get("clarification_context", None)

            _reset_turn_fields(state)

            state["input"] = user_input
            state["chat_history"] = chat_history
            if session_id is not None:
                state["session_id"] = session_id
            state["clarification_context"] = clarification_ctx
            state["retry_count"] = 0
            state["retry_type"] = None

            if state.get("debug_info") is None or not isinstance(state.get("debug_info"), dict):
                state["debug_info"] = {}

            # chat_history를 텍스트로 변환
            context_lines = []
            for msg in chat_history:
                content = getattr(msg, "content", None)
                content = content if isinstance(content, str) else str(msg)
                msg_type = getattr(msg, "type", None)
                if msg_type == "human":
                    context_lines.append(f"사용자: {content}")
                elif msg_type == "ai":
                    context_lines.append(f"어시스턴트: {content}")
                else:
                    context_lines.append(content)
            context_text = "\n".join(context_lines).strip()
            state['previous_context'] = context_text

            # dict_hint 생성
            dict_hint = infer_dict_hint(user_input, context_text=context_text, rag_dict_index=rag_dict_index)
            state["dict_hint"] = dict_hint
            allowed = {"SMALLTALK", "META", "RAG", "GENERAL_ADVICE"}

            # 라우터 실행
            intent_raw = ""
            router_output = None
            try:
                router_chain = router_prompt | router_llm | StrOutputParser()
                router_output = router_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history,
                    "chat_history_text": context_text,
                })
                intent_raw = _norm_label(router_output)
            except Exception as _e:
                intent_raw = ""

            if intent_raw in allowed:
                intent = intent_raw
            else:
                intent = "RAG"

            # 개인 기억 질문 체크
            if context_text and is_personal_memory_question(context=context_text, curr=user_input):
                intent = "SMALLTALK"

            # RAG 오버라이드 판정
            if intent != "RAG" and context_text:
                rag_override_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "질문이 '통계 보고서 데이터'를 필요로 하는지 판단합니다.\n"
                     "현재 질문이 보고서의 수치/분석/비교를 필요로 하면 YES, 아니면 NO."
                    ),
                    ("human",
                     "[이전 대화]\n{context}\n\n[현재 질문]\n{question}\n\n판정:")
                ])
                try:
                    override_chain = rag_override_prompt | router_llm | StrOutputParser()
                    override_result = override_chain.invoke({
                        "context": context_text[-2000:],
                        "question": user_input,
                    }).strip().upper()
                    if "YES" in override_result:
                        intent = "RAG"
                except Exception:
                    pass

            state['intent_raw'] = intent_raw
            state['intent'] = intent

            state['is_chat_reference'] = False
            state['followup_type'] = None
            state['resolved_question'] = user_input

            if intent == "RAG":
                is_ref = is_chat_reference_question(context=context_text, curr=user_input)
                state["is_chat_reference"] = bool(is_ref)

                if is_ref:
                    state['followup_type'] = "rag_standalone_rewrite"
                    resolved_q = classify_followup_type(user_input=user_input, context=context_text)
                    state['resolved_question'] = resolved_q.strip()
                else:
                    state['followup_type'] = None
                    state["resolved_question"] = user_input
            else:
                state['is_chat_reference'] = None
                state['followup_type'] = f"{intent.lower()}_full_context"
                q = user_input.strip()
                if context_text:
                    state['resolved_question'] = f"[이전대화]\n{context_text}\n\n[현재질문]\n{q}"
                else:
                    state['resolved_question'] = q

            state["debug_info"]["route_intent"] = {
                "intent_raw": intent_raw,
                "intent_final": intent,
                "user_input": user_input,
                "resolved_question": state.get("resolved_question"),
                "followup_type": state.get("followup_type"),
                "is_chat_reference": state.get("is_chat_reference"),
            }
            return state

        except Exception as e:
            state['intent'] = "META"
            state['resolved_question'] = state.get("input", "")
            return state

    def respond_smalltalk(state: GraphState) -> GraphState:
        """SMALLTALK intent 응답 생성."""
        status_callback("💬 응답 생성 중...")
        try:
            user_input = state.get("input", "")
            chat_history = state.get('chat_history', [])

            smalltalk_prompt = ChatPromptTemplate.from_messages([
                ("system", smalltalk_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            smalltalk_chain = smalltalk_prompt | casual_llm | StrOutputParser()
            answer = smalltalk_chain.invoke({'input': user_input, 'chat_history': chat_history})
            answer = (answer or "").strip()

            state['draft_answer'] = answer
            state['final_answer'] = answer
            return state

        except Exception:
            state['final_answer'] = "질문을 다시한번만 입력해주십시오."
            return state

    def respond_meta(state: GraphState) -> GraphState:
        """META intent 응답 생성."""
        status_callback("ℹ️ 시스템 정보 제공 중...")
        try:
            user_input = state.get('input', "")
            chat_history = state.get("chat_history", [])

            meta_prompt = ChatPromptTemplate.from_messages([
                ("system", meta_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            meta_chain = meta_prompt | casual_llm | StrOutputParser()
            answer = meta_chain.invoke({'input': user_input, "chat_history": chat_history})
            answer = (answer or "").strip()

            state['draft_answer'] = answer
            state['final_answer'] = answer
            state['formatted_answer'] = answer
            return state

        except Exception:
            state['final_answer'] = "질문을 다시한번만 입력해주십시오."
            return state

    def respond_general_advice(state: GraphState) -> GraphState:
        """GENERAL_ADVICE intent 응답 생성."""
        status_callback("💡 조언 생성 중...")
        try:
            user_input = state.get('input', "")
            chat_history = state.get("chat_history", [])

            general_advice_prompt = ChatPromptTemplate.from_messages([
                ("system", general_advice_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            general_advice_chain = general_advice_prompt | casual_llm | StrOutputParser()
            answer = general_advice_chain.invoke({'input': user_input, "chat_history": chat_history})
            answer = (answer or "").strip()

            state['draft_answer'] = answer
            state['final_answer'] = answer
            state['formatted_answer'] = answer
            return state

        except Exception:
            state['final_answer'] = "일반적인 조언 생성 중 오류가 발생했습니다."
            return state

    def plan_search(state: GraphState) -> GraphState:
        """검색 계획을 수립한다."""
        status_callback("📋 검색 계획 수립 중...")
        try:
            user_input = (state.get("resolved_question") or state.get("input") or "").strip()
            chat_history = state.get("chat_history", [])

            dict_hint = state.get("dict_hint") or infer_dict_hint(
                user_input,
                context_text=state.get("previous_context", ""),
                rag_dict_index=rag_dict_index
            )
            state["dict_hint"] = dict_hint

            raw_followup_type = state.get("followup_type") or "none"
            followup_type = raw_followup_type

            history_hints = _extract_last_context_hints(chat_history)
            history_years = history_hints.get("years", [])

            if followup_type == "rag_standalone_rewrite":
                if history_years:
                    followup_type = "detail_request"
                else:
                    followup_type = "none"

            if followup_type == "none":
                topic_core = ""
                last_target = ""
                last_years = []
            else:
                topic_core = state.get("last_topic_core", "") or ""
                last_target = state.get("last_target", "") or ""
                last_years = state.get("last_years", []) or []

                if not last_years and history_years:
                    last_years = history_years

                if not topic_core and history_hints.get("last_user_query"):
                    topic_core = history_hints["last_user_query"]

            if not last_target and dict_hint.get("target_group"):
                last_target = dict_hint["target_group"]

            if not topic_core:
                tc = dict_hint.get("topic_code") or ""
                at = (dict_hint.get("anchor_terms") or [])
                anchor_short = at[0] if at else ""
                topic_core = f"{tc} {anchor_short}".strip()

            if state.get("debug_info") is None:
                state["debug_info"] = {}

            planner_prompt = ChatPromptTemplate.from_messages([
                ("system", planner_prompt_text),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            planner_chain = planner_prompt | validator_llm | StrOutputParser()

            effective_history = []
            if followup_type != "none":
                effective_history = chat_history[-4:] if len(chat_history) > 4 else chat_history

            result = planner_chain.invoke({
                "input": user_input,
                "chat_history": effective_history,
            })

            json_match = re.search(r'\{[\s\S]*\}', result or "")
            if json_match:
                result = json_match.group()
            plan = json.loads(result)

            # 연도 결정 로직
            validator_years = plan.get("years", [])
            if not isinstance(validator_years, list):
                validator_years = []
            validator_years = sorted([
                y for y in validator_years
                if isinstance(y, int) and y in YEAR_TO_FILENAME
            ])

            input_years = parse_year_range(user_input)
            input_years = sorted([
                y for y in input_years
                if isinstance(y, int) and y in YEAR_TO_FILENAME
            ])

            if input_years:
                years = input_years
                year_source = "parse_year_range"
            elif followup_type != "none" and history_years:
                years = sorted([
                    y for y in history_years
                    if isinstance(y, int) and y in YEAR_TO_FILENAME
                ])
                year_source = "history_year_hint"
            elif validator_years:
                years = validator_years
                year_source = "validator_years"
            else:
                years = []
                year_source = "empty"

            state["debug_info"].setdefault("plan_search", {})
            state["debug_info"]["plan_search"]["year_source"] = year_source
            state["debug_info"]["plan_search"]["validator_years_filtered"] = validator_years
            state["debug_info"]["plan_search"]["input_years_parsed"] = input_years

            # file_name_filters 정합성 보정
            fns = plan.get("file_name_filters", [])
            if not isinstance(fns, list):
                fns = []
            fns = [fn for fn in fns if isinstance(fn, str) and fn in ALLOWED_FILES]

            if years and not fns:
                fns = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]

            if years:
                expected = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]
                if set(fns) != set(expected):
                    fns = expected

            # years가 없으면 디폴트 지정
            used_default_years = False
            if not years:
                years = [2023, 2024]
                used_default_years = True
            state["used_default_years"] = used_default_years

            # [개선] 연도 수에 따른 동적 쿼리 수 계산
            num_years = len(years)
            params = calculate_search_params(num_years)
            max_queries = params["max_queries"]

            # queries 정리
            queries = plan.get("queries", [])
            if not isinstance(queries, list):
                queries = []
            queries = [str(q).strip() for q in queries if str(q).strip()]

            resolved_q = plan.get("resolved_question", user_input)
            if not isinstance(resolved_q, str) or not resolved_q.strip():
                resolved_q = user_input
            resolved_q = resolved_q.strip()

            # [개선] 연도별 쿼리 보장
            base_query_clean = re.sub(r'20[2][0-4]년?\s*', '', resolved_q).strip()
            base_query_clean = re.sub(r'20[2][0-4]~?20[2][0-4]년?\s*', '', base_query_clean).strip()
            
            for y in years:
                year_query = f"{y}년 {base_query_clean}"
                if year_query not in queries:
                    queries.append(year_query)

            while len(queries) < 3:
                queries.append(resolved_q)
            queries = queries[:max_queries]

            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "queries": queries,
                "resolved_question": resolved_q,
                "followup_type": followup_type,
                "followup_type_raw": raw_followup_type,
                "topic_code": dict_hint.get("topic_code", ""),
                "target_group": dict_hint.get("target_group", ""),
                "dict_hint": dict_hint,
                "used_default_years": used_default_years,
            }
            state["resolved_question"] = resolved_q

            state["debug_info"]["plan_search"].update({
                "input_used": user_input,
                "final_years": years,
                "final_files": fns,
                "final_queries": queries,
                "final_resolved_question": resolved_q,
            })
            return state

        except Exception as e:
            logger.warning("플래너 에러: %s", e)
            user_input = (state.get("resolved_question") or state.get("input") or "").strip()
            years = parse_year_range(user_input)

            if not years and state.get("chat_history"):
                years = _extract_years_from_chat_history(state["chat_history"])

            fns = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]

            state["plan"] = {
                "years": years if years else [2023, 2024],
                "file_name_filters": fns,
                "queries": [user_input] * 3,
                "resolved_question": user_input,
                "followup_type": "none",
            }
            state["resolved_question"] = user_input
            return state

    def query_rewrite(state: GraphState) -> GraphState:
        """
        [개선] 검색 쿼리를 LLM으로 최적화한다 - 연도별 균등 분배.
        """
        status_callback("🔧 쿼리 최적화 중...")
        try:
            plan = state["plan"]
            queries = plan.get("queries", [])
            resolved_q = plan.get("resolved_question", "")
            years = plan.get("years", [])
            dict_hint = state.get("dict_hint") or {}

            # [개선] 연도별 쿼리 추가 (멀티연도 시)
            base_query_clean = re.sub(r'20[2][0-4]년?\s*', '', resolved_q).strip()
            base_query_clean = re.sub(r'20[2][0-4]~?20[2][0-4]년?\s*', '', base_query_clean).strip()
            
            if len(years) > 1:
                for y in years:
                    year_query = f"{y}년 {base_query_clean}"
                    if year_query not in queries:
                        queries.append(year_query)

            # LLM 기반 최적화 수행
            result = (_rewrite_prompt_25 | rewrite_llm | StrOutputParser()).invoke({
                "resolved_question": resolved_q,
                "queries": str(queries),
                "years": str(years),
            })

            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()

            optimized = json.loads(result)
            rewritten = optimized.get("optimized_queries", queries)

            if not isinstance(rewritten, list) or not rewritten:
                rewritten = queries

            unique_queries = list(dict.fromkeys(rewritten))

            # [개선] 연도별 쿼리 분포 확인 및 보완
            year_query_count = {y: 0 for y in years}
            
            for q in unique_queries:
                for y in years:
                    if str(y) in q or f"{y}년" in q:
                        year_query_count[y] += 1
                        break
            
            # 부족한 연도에 쿼리 추가
            for y in years:
                deficit = MIN_QUERIES_PER_YEAR - year_query_count[y]
                for i in range(deficit):
                    if i == 0:
                        new_query = f"{y}년 {base_query_clean}"
                    else:
                        # RAG Dictionary 기반 키워드 활용
                        keywords = extract_keywords_from_dict(resolved_q, rag_dict_index, dict_hint)
                        kw_str = " ".join(keywords[:2])
                        new_query = f"{y}년 {kw_str}"
                    
                    if new_query not in unique_queries:
                        unique_queries.append(new_query)

            # 앵커 용어로 쿼리 보강
            anchors = dict_hint.get("anchor_terms", [])
            if anchors:
                unique_queries = augment_queries_with_anchors(unique_queries, anchors)

            # [개선] 동적 쿼리 수 상한 계산
            num_years = len(years) if years else 2
            params = calculate_search_params(num_years)
            max_queries = params["max_queries"]

            state["rewritten_queries"] = unique_queries[:max_queries]
            state["plan"]["queries"] = unique_queries[:max_queries]
            
            # 디버그 정보
            state.setdefault("debug_info", {})
            state["debug_info"]["query_year_distribution"] = year_query_count
            
            return state

        except Exception as e:
            state["rewritten_queries"] = state["plan"].get("queries", [])
            return state

    def retrieve_documents(state: GraphState) -> GraphState:
        """
        [개선] ChromaDB에서 관련 문서를 검색한다 - 연도별 균등 검색.
        
        핵심 개선:
        1. 연도별 독립 검색 수행
        2. Round-robin 방식으로 균등한 parent ID 선정
        3. 연도별 최소 확보량 보장
        """
        retry_count = state.get("retry_count", 0)
        retry_info = f" (재시도 #{retry_count})" if retry_count > 0 else ""
        status_callback(f"🔍 보고서 검색 중...{retry_info}")

        try:
            plan = state["plan"]
            target_files = plan.get("file_name_filters", [])
            target_years = plan.get("years", [])
            queries = state.get("rewritten_queries") or plan.get("queries", [])
            resolved_q = plan.get("resolved_question", "")
            dict_hint = state.get("dict_hint") or {}

            # [개선 1] 동적 파라미터 계산
            num_years = len(target_years) if target_years else 2
            params = calculate_search_params(num_years, retry_count)
            
            k_per_query = params["k_per_query"]
            top_parents = params["top_parents"]
            top_parents_per_file = params["top_parents_per_file"]

            # [개선 2] 연도별 독립 검색 수행
            year_to_docs: Dict[int, List[Document]] = {}
            
            for year in target_years:
                fn = YEAR_TO_FILENAME.get(year)
                if not fn:
                    continue
                
                file_filter = {'$and': [
                    {'doc_type': {"$in": SUMMARY_TYPES}},
                    {'file_name': fn}
                ]}
                
                year_docs = []
                seen_keys = set()
                
                # 해당 연도 전용 쿼리 생성
                year_specific_queries = generate_year_specific_queries(
                    queries, year, resolved_q, rag_dict_index, dict_hint
                )
                
                # 검색 수행
                for q in year_specific_queries:
                    if not q:
                        continue
                    try:
                        hits = vectorstore.similarity_search_with_relevance_scores(
                            q, k=k_per_query, filter=file_filter
                        )
                        for doc, score in hits:
                            key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                            if key not in seen_keys:
                                doc.metadata["_score"] = float(score)
                                doc.metadata["_year"] = year
                                doc.metadata["_source_file"] = fn
                                year_docs.append(doc)
                                seen_keys.add(key)
                    except Exception:
                        pass
                
                # 키워드 부스트 적용
                for doc in year_docs:
                    boost = _keyword_boost_score(doc, resolved_q, dict_hint)
                    doc.metadata["_final_score"] = doc.metadata.get("_score", 0) + boost
                
                # 정렬
                year_docs.sort(key=lambda d: d.metadata.get("_final_score", 0), reverse=True)
                year_to_docs[year] = year_docs

            # [개선 3] Round-robin 방식 Parent ID 선정
            parent_ids = []
            seen_pid = set()
            
            # 각 연도에서 최소 MIN_PARENTS_PER_YEAR개 확보 목표
            max_rounds = max(
                (len(docs) for docs in year_to_docs.values()),
                default=0
            )
            
            for round_idx in range(max_rounds):
                if len(parent_ids) >= top_parents:
                    break
                for year in target_years:
                    if len(parent_ids) >= top_parents:
                        break
                    year_docs = year_to_docs.get(year, [])
                    if round_idx < len(year_docs):
                        doc = year_docs[round_idx]
                        pid = doc.metadata.get("parent_id")
                        if pid and pid not in seen_pid:
                            parent_ids.append(pid)
                            seen_pid.add(pid)

            # [개선 4] 연도별 최소 확보 검증
            year_parent_count = {y: 0 for y in target_years}
            for pid in parent_ids:
                for year, docs in year_to_docs.items():
                    if any(d.metadata.get("parent_id") == pid for d in docs):
                        year_parent_count[year] += 1
                        break

            # Chunk 확장
            all_docs = []
            for docs in year_to_docs.values():
                all_docs.extend(docs)

            expanded_chunks = []
            for pid in parent_ids:
                try:
                    got = vectorstore._collection.get(
                        where={'parent_id': pid},
                        include=['documents', 'metadatas']
                    )
                    chunks = []
                    for txt, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                        if isinstance(meta, dict) and meta.get("doc_type") == "text_chunk":
                            chunks.append((int(meta.get("chunk_index", 0)), txt or "", meta))

                    chunks.sort(key=lambda x: x[0])
                    for _, txt, meta in chunks[:MAX_CHUNKS_PER_PARENT]:
                        expanded_chunks.append(Document(page_content=txt, metadata=meta))
                except Exception:
                    pass

            pid_set = set(parent_ids)
            kept_summaries = [d for d in all_docs if d.metadata.get("parent_id") in pid_set]
            final_docs = kept_summaries + expanded_chunks

            # 컨텍스트 블록 구성
            blocks = []
            for i, d in enumerate(final_docs, start=1):
                m = d.metadata
                text = d.page_content[:MAX_CHARS_PER_DOC]
                blocks.append(f"[{i}] {m.get('file_name', 'unknown')} (p.{m.get('page', '?')})\n{text}")

            files_searched = list(set(
                d.metadata.get("file_name") for d in all_docs 
                if d.metadata.get("file_name")
            ))

            state["retrieval"] = {
                "docs": final_docs,
                "parent_ids": parent_ids,
                "files_searched": files_searched,
                "doc_count": len(final_docs),
                "year_distribution": year_parent_count,
            }
            state["context"] = "\n\n---\n\n".join(blocks)
            
            # 디버그 정보
            state.setdefault("debug_info", {})
            state["debug_info"]["year_parent_distribution"] = year_parent_count

            return state

        except Exception as e:
            state["context"] = ""
            state["retrieval"] = {
                "docs": [], "parent_ids": [], 
                "files_searched": [], "doc_count": 0,
                "error": str(e)
            }
            return state

    def rerank_compress(state: GraphState) -> GraphState:
        """
        [개선] 검색 결과를 리랭킹하고 압축한다 - 연도별 균등 배치.
        """
        status_callback("📊 결과 정렬 및 압축 중...")
        try:
            docs = state.get("retrieval", {}).get("docs", [])
            query = state.get("resolved_question", "")
            years = state.get("plan", {}).get("years", [])
            dict_hint = state.get("dict_hint") or {}

            if not docs:
                state["reranked_docs"] = []
                state["compressed_context"] = ""
                return state

            query_keywords = set(re.findall(r'[가-힣]+', query))

            for doc in docs:
                content_keywords = set(re.findall(r'[가-힣]+', doc.page_content or ""))
                overlap = len(query_keywords & content_keywords)
                doc.metadata["_rerank_score"] = doc.metadata.get("_final_score", 0) + (overlap * 0.01)

            # [개선 1] 연도별 문서 그룹화
            year_to_docs: Dict[int, List[Document]] = {y: [] for y in years}
            other_docs = []
            
            for doc in docs:
                doc_year = extract_year_from_doc(doc)
                if doc_year and doc_year in year_to_docs:
                    year_to_docs[doc_year].append(doc)
                else:
                    other_docs.append(doc)
            
            # 각 연도별 내부 정렬
            for y in years:
                year_to_docs[y].sort(
                    key=lambda d: d.metadata.get("_rerank_score", 0),
                    reverse=True
                )

            # 중복 제거
            seen_content = set()
            
            def dedupe(doc_list: List[Document]) -> List[Document]:
                nonlocal seen_content
                result = []
                for doc in doc_list:
                    content_hash = hash(doc.page_content[:500])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        result.append(doc)
                return result
            
            for y in years:
                year_to_docs[y] = dedupe(year_to_docs[y])
            other_docs = dedupe(other_docs)

            # [개선 2] Round-robin 방식으로 균등 배치
            compressed_docs = []
            
            for round_idx in range(MAX_DOCS_PER_YEAR_IN_CONTEXT):
                for y in years:
                    if round_idx < len(year_to_docs[y]):
                        compressed_docs.append(year_to_docs[y][round_idx])

            # 나머지 문서 추가 (최대 20개까지)
            remaining_space = 20 - len(compressed_docs)
            if remaining_space > 0:
                compressed_docs.extend(other_docs[:remaining_space])

            compressed_docs = compressed_docs[:20]

            # [신규] 유사 내용 탐색
            similar_info = find_similar_available_content(query, context_text, rag_dict_index)
            if similar_info.get("has_similar"):
                state["similar_content_info"] = similar_info
                state.setdefault("debug_info", {})
                state["debug_info"]["similar_content"] = similar_info


            blocks = []
            for i, d in enumerate(compressed_docs, start=1):
                m = d.metadata
                text = d.page_content[:MAX_CHARS_PER_DOC]
                blocks.append(f"[{i}] {m.get('file_name', 'unknown')} (p.{m.get('page', '?')})\n{text}")

            state["reranked_docs"] = compressed_docs
            state["compressed_context"] = "\n\n---\n\n".join(blocks)
            
            # 디버그 정보
            year_doc_count = {y: len(docs) for y, docs in year_to_docs.items()}
            state.setdefault("debug_info", {})
            state["debug_info"]["compressed_year_distribution"] = year_doc_count

            return state

        except Exception as e:
            state["reranked_docs"] = state.get("retrieval", {}).get("docs", [])
            state["compressed_context"] = state.get("context", "")
            return state

    def extract_key_figures(state: GraphState) -> GraphState:
        """
        [개선] 다중 연도 핵심 수치를 사전 추출한다 - 누락 연도 재검색 추가.
        """
        plan = state.get("plan") or {}
        years = plan.get("years", [])

        if len(years) <= 1:
            return state

        context = (state.get("compressed_context") or state.get("context", ""))
        resolved_q = (state.get("resolved_question") or state.get("input", ""))

        if not context.strip():
            return state

        status_callback("📈 핵심 수치 추출 중...")
        try:
            years_str = ", ".join([str(y) for y in years if str(y).strip()])
            resolved_q_for_extract = f"{resolved_q}\n[요청 연도] {years_str}".strip()

            raw = (EXTRACT_FIGURES_PROMPT | rewrite_llm | StrOutputParser()).invoke({
                "resolved_question": resolved_q_for_extract,
                "context": context[:20000],
            })

            parsed = _safe_parse_json(raw)

            rows = []
            if parsed and isinstance(parsed.get("연도별_수치"), list):
                rows = [r for r in parsed["연도별_수치"] if isinstance(r, dict)]
            if not rows:
                return state

            required_years = []
            for y in years:
                try:
                    required_years.append(int(str(y).strip()))
                except Exception:
                    continue
            required_years = sorted(list(dict.fromkeys(required_years)))

            by_year = {}
            for row in rows:
                try:
                    yy = int(str(row.get("연도", "")).strip())
                except Exception:
                    continue
                if yy in required_years and yy not in by_year:
                    by_year[yy] = row

            def _is_na(v):
                s = str(v).strip()
                if not s:
                    return True
                return s.upper() in {"N/A", "NA"} or s in {"-", "없음", "미제시", "해당없음"}

            missing_years = []
            for yy in required_years:
                row = by_year.get(yy)
                if not row:
                    missing_years.append(yy)
                    continue
                vals = [row.get(k, "N/A") for k in ["전체", "유아동", "청소년", "성인", "60대"]]
                if all(_is_na(v) for v in vals):
                    missing_years.append(yy)

            # [개선] 누락 연도가 있으면 타겟 재검색 수행
            if missing_years and len(missing_years) < len(years):
                status_callback(f"🔄 누락 연도 {missing_years} 추가 검색 중...")
                
                dict_hint = state.get("dict_hint") or {}
                supplemental_context = targeted_year_search(
                    missing_years, 
                    resolved_q,
                    vectorstore,
                    rag_dict_index,
                    dict_hint
                )
                
                if supplemental_context:
                    # 보완 컨텍스트로 재추출 시도
                    combined_context = f"{context}\n\n---\n\n[보완 검색 결과]\n{supplemental_context}"
                    
                    re_raw = (EXTRACT_FIGURES_PROMPT | rewrite_llm | StrOutputParser()).invoke({
                        "resolved_question": resolved_q_for_extract,
                        "context": combined_context[:25000],
                    })
                    
                    re_parsed = _safe_parse_json(re_raw)
                    if re_parsed and isinstance(re_parsed.get("연도별_수치"), list):
                        for row in re_parsed["연도별_수치"]:
                            if isinstance(row, dict):
                                try:
                                    yy = int(str(row.get("연도", "")).strip())
                                    if yy in missing_years and yy not in by_year:
                                        by_year[yy] = row
                                except Exception:
                                    continue

            summary_lines = ["[연도별 핵심 수치 요약 — 아래 수치를 우선 참조하십시오]"]
            summary_lines.append("| 연도 | 전체 | 유아동 | 청소년 | 성인 | 60대 |")
            summary_lines.append("|------|------|--------|--------|------|------|")

            ordered_rows = []
            for yy in required_years:
                row = by_year.get(yy) or {"연도": yy}
                ordered_row = {
                    "연도": yy,
                    "전체": row.get("전체", "N/A"),
                    "유아동": row.get("유아동", "N/A"),
                    "청소년": row.get("청소년", "N/A"),
                    "성인": row.get("성인", "N/A"),
                    "60대": row.get("60대", "N/A"),
                }
                ordered_rows.append(ordered_row)
                summary_lines.append(
                    f"| {yy} | {ordered_row['전체']} | {ordered_row['유아동']} | "
                    f"{ordered_row['청소년']} | {ordered_row['성인']} | {ordered_row['60대']} |"
                )

            state["extracted_figures"] = "\n".join(summary_lines)
            state["extracted_figures_json"] = {"연도별_수치": ordered_rows}
            state["year_extractions"] = ordered_rows

            state.setdefault("debug_info", {})
            state["debug_info"]["missing_years"] = missing_years

        except Exception as e:
            pass

        return state

    def context_sanitize(state: GraphState) -> GraphState:
        """컨텍스트에서 프롬프트 인젝션 패턴을 제거한다."""
        status_callback("🛡️ 컨텍스트 검증 중...")
        try:
            base_context = state.get("compressed_context") or state.get("context", "")

            extracted = state.get("extracted_figures", "")
            if extracted and base_context.strip():
                combined_context = f"{extracted}\n\n---\n\n{base_context}"
            else:
                combined_context = base_context

            danger_patterns = [
                r"(?i)ignore\s+(previous|above|all)\s+instructions?",
                r"(?i)you\s+are\s+now\s+",
                r"(?i)act\s+as\s+",
                r"(?i)system\s*:\s*",
            ]

            sanitized = combined_context
            for pattern in danger_patterns:
                sanitized = re.sub(pattern, "[FILTERED]", sanitized)

            state["sanitized_context"] = sanitized
            return state

        except Exception:
            base_context = state.get("compressed_context") or state.get("context", "")
            extracted = state.get("extracted_figures", "")
            state["sanitized_context"] = f"{extracted}\n\n---\n\n{base_context}" if extracted and base_context.strip() else base_context
            return state

    def generate_answer(state: GraphState) -> GraphState:
        """LLM을 사용하여 최종 답변을 생성한다."""
        retry_count = state.get("retry_count", 0)
        retry_info = f" (재생성 #{retry_count})" if retry_count > 0 and state.get("retry_type") == "generate" else ""
        status_callback(f"✍️ 답변 생성 중...{retry_info}")

        try:
            context = state.get("sanitized_context") or state.get("compressed_context") or state.get("context", "")

            if not context.strip():
                state["draft_answer"] = "검색 결과를 찾지 못했습니다. 질문을 다시 구체적으로 말씀해주시겠습니까?"
                return state

            dict_hint = state.get("dict_hint") or {}
            resolved_q = state.get("resolved_question") or state.get("input", "")
            context_guard = build_context_guard(dict_hint, resolved_q)

            if retry_count > 0 and state.get("retry_type") == "generate":
                previous_issue = state.get("validation_reason", "형식 문제")
                answer = (_answer_retry_prompt_25 | main_llm | StrOutputParser()).invoke({
                    "input": resolved_q,
                    "context": context,
                    "previous_issue": previous_issue,
                    "context_guard": context_guard,
                })
            else:
                answer = (_answer_prompt_25 | main_llm | StrOutputParser()).invoke({
                    "input": resolved_q,
                    "context": context,
                    "context_guard": context_guard,
                })

            state["draft_answer"] = answer
            return state

        except Exception as e:
            state["draft_answer"] = f"답변 생성 중 오류: {e}"
            return state

    def safety_check(state: GraphState) -> GraphState:
        """답변에 민감한 패턴이 있는지 검사한다."""
        status_callback("🔒 안전성 검사 중...")
        try:
            answer = state.get("draft_answer", "")
            issues = []

            sensitive_patterns = [
                (r"(?i)(자살|자해)", "자해 관련 내용"),
                (r"(?i)(폭력|학대)", "폭력 관련 내용"),
            ]

            for pattern, issue_name in sensitive_patterns:
                if re.search(pattern, answer):
                    issues.append(issue_name)

            state["safety_passed"] = len(issues) == 0
            state["safety_issues"] = issues
            return state

        except Exception:
            state["safety_passed"] = True
            state["safety_issues"] = []
            return state

    def validate_answer(state: GraphState) -> GraphState:
        """LLM으로 답변 품질을 검증한다."""
        status_callback("✅ 답변 검증 중...")
        try:
            retry_count = state.get("retry_count", 0)

            if retry_count >= MAX_RETRY_COUNT:
                state["validation_result"] = "PASS"
                final_answer = state["draft_answer"]

                if state.get("used_default_years"):
                    final_answer = _append_year_confirmation(final_answer, state)

                state["final_answer"] = final_answer
                return state

            context = state.get("sanitized_context") or state.get("context", "")

            dict_hint = state.get("dict_hint") or {}
            resolved_q = state.get("resolved_question") or state.get("input", "")
            context_guard = build_context_guard(dict_hint, resolved_q)

            result = (_validator_prompt_25 | validator_llm | StrOutputParser()).invoke({
                "input": resolved_q,
                "context": context[:15000],
                "answer": state["draft_answer"],
                "context_guard": context_guard,
            })

            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()

            validator_out = json.loads(result)
            state["validator_output"] = validator_out

            validation_result = validator_out.get("result", "PASS").upper()
            valid_results = ["PASS", "FAIL_NO_EVIDENCE", "FAIL_UNCLEAR", "FAIL_FORMAT"]
            if validation_result not in valid_results:
                validation_result = "PASS"

            state["validation_result"] = validation_result
            state["validation_reason"] = validator_out.get("reason", "")

            if validation_result == "PASS":
                corrected = validator_out.get("corrected_answer", "")
                final_answer = corrected if corrected and len(corrected) > 50 else state["draft_answer"]
                if state.get("used_default_years"):
                    final_answer = _append_year_confirmation(final_answer, state)
                state["final_answer"] = final_answer

            elif validation_result == "FAIL_UNCLEAR":
                clarify_q = validator_out.get("clarify_question", "")
                if clarify_q:
                    state["pending_clarification"] = clarify_q

            scope_issues = detect_scope_mismatch(
                state.get("draft_answer", ""),
                context,
                dict_hint
            )

            if scope_issues and validation_result == "PASS":
                state["validation_reason"] = "; ".join(scope_issues)

            return state

        except Exception as e:
            state["validation_result"] = "PASS"
            final_answer = state.get("draft_answer", "")
            if state.get("used_default_years"):
                final_answer = _append_year_confirmation(final_answer, state)
            state["final_answer"] = final_answer
            return state

    def handle_clarify(state: GraphState) -> GraphState:
        """질문 명확화가 필요할 때 추가 질문을 생성한다."""
        status_callback("❓ 명확화 질문 생성 중...")
        try:
            clarify_question = state.get("pending_clarification", "")
            if not clarify_question:
                clarify_question = (
                    "질문을 좀 더 구체적으로 말씀해 주시겠습니까? "
                    "예를 들어, 특정 연도나 대상(청소년, 성인 등)을 지정해 주시면 "
                    "더 정확한 답변이 가능합니다."
                )

            state["clarification_context"] = {
                "original_query": state["input"],
                "partial_plan": state.get("plan"),
            }
            state["final_answer"] = clarify_question
            return state

        except Exception:
            state["final_answer"] = "질문을 좀 더 구체적으로 말씀해 주시겠습니까?"
            return state

    def retrieve_retry(state: GraphState) -> GraphState:
        """검색 재시도 시 쿼리를 확장한다."""
        status_callback("🔄 검색 재시도 준비 중...")
        state["retry_count"] = (state.get("retry_count") or 0) + 1
        state["retry_type"] = "retrieve"

        queries = state["plan"].get("queries", [])
        resolved_q = state.get("resolved_question", "")
        dict_hint = state.get("dict_hint") or {}

        # [개선] RAG Dictionary 기반 키워드로 쿼리 확장
        keywords = extract_keywords_from_dict(resolved_q, rag_dict_index, dict_hint)
        
        expanded_queries = list(queries)
        for kw in keywords[:3]:
            new_query = f"{resolved_q} {kw}"
            if new_query not in expanded_queries:
                expanded_queries.append(new_query)

        # 연도별 쿼리 추가
        years = state["plan"].get("years", [])
        base_query_clean = re.sub(r'20[2][0-4]년?\s*', '', resolved_q).strip()
        for y in years:
            year_query = f"{y}년 {base_query_clean}"
            if year_query not in expanded_queries:
                expanded_queries.append(year_query)

        state["plan"]["queries"] = expanded_queries[:12]
        state["rewritten_queries"] = expanded_queries[:12]
        return state

    def generate_retry(state: GraphState) -> GraphState:
        """답변 재생성 시 카운터를 증가시킨다."""
        status_callback("🔄 답변 재생성 준비 중...")
        state["retry_count"] = (state.get("retry_count") or 0) + 1
        state["retry_type"] = "generate"
        return state

    # 모든 노드 함수 반환
    return {
        "route_intent": route_intent,
        "smalltalk": respond_smalltalk,
        "meta": respond_meta,
        "general_advice": respond_general_advice,
        "plan_search": plan_search,
        "query_rewrite": query_rewrite,
        "retrieve": retrieve_documents,
        "rerank_compress": rerank_compress,
        "extract_key_figures": extract_key_figures,
        "context_sanitize": context_sanitize,
        "generate": generate_answer,
        "safety_check": safety_check,
        "validate": validate_answer,
        "clarify": handle_clarify,
        "retrieve_retry": retrieve_retry,
        "generate_retry": generate_retry,
    }


# =========================================================
# 그래프 빌드
# =========================================================
def build_graph(node_functions):
    """
    LangGraph 워크플로우를 구성하고 컴파일한다.
    
    Args:
        node_functions: create_node_functions에서 반환된 노드 함수 딕셔너리
        
    Returns:
        CompiledGraph: 컴파일된 LangGraph 워크플로우
    """
    workflow = StateGraph(GraphState)

    # 노드 등록
    for name, func in node_functions.items():
        workflow.add_node(name, func)

    # 라우팅 함수
    def route_by_intent(state: GraphState) -> str:
        """intent에 따른 노드 분기."""
        intent = state.get("intent", "RAG")
        if intent == "SMALLTALK":
            return "smalltalk"
        elif intent == "META":
            return "meta"
        elif intent == "GENERAL_ADVICE":
            return "general_advice"
        else:
            return "rag_pipeline"

    def route_after_validate(state: GraphState) -> str:
        """Validation 결과에 따른 분기."""
        retry_count = state.get("retry_count", 0)
        if retry_count >= MAX_RETRY_COUNT:
            return "end"
        result = state.get("validation_result", "PASS")
        if result == "PASS":
            return "end"
        elif result == "FAIL_NO_EVIDENCE":
            return "retrieve_retry"
        elif result == "FAIL_UNCLEAR":
            return "clarify"
        elif result == "FAIL_FORMAT":
            return "generate_retry"
        else:
            return "end"

    # Entry point
    workflow.set_entry_point("route_intent")

    # Intent 분기
    workflow.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {
            "smalltalk": "smalltalk",
            "meta": "meta",
            "general_advice": "general_advice",
            "rag_pipeline": "plan_search"
        }
    )

    # 종료 노드
    workflow.add_edge("smalltalk", END)
    workflow.add_edge("meta", END)
    workflow.add_edge("general_advice", END)
    workflow.add_edge("clarify", END)

    # RAG 파이프라인
    workflow.add_edge("plan_search", "query_rewrite")
    workflow.add_edge("query_rewrite", "retrieve")
    workflow.add_edge("retrieve", "rerank_compress")
    workflow.add_edge("rerank_compress", "extract_key_figures")
    workflow.add_edge("extract_key_figures", "context_sanitize")
    workflow.add_edge("context_sanitize", "generate")
    workflow.add_edge("generate", "safety_check")
    workflow.add_edge("safety_check", "validate")

    # Validation 후 분기
    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "end": END,
            "retrieve_retry": "retrieve_retry",
            "clarify": "clarify",
            "generate_retry": "generate_retry"
        }
    )

    # 재시도 루프
    workflow.add_edge("retrieve_retry", "retrieve")
    workflow.add_edge("generate_retry", "generate")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

