# =========================================================
# Streamlit 기반 스마트폰 과의존 실태조사 RAG 챗봇 v5.2
# 
# LangGraph 로직은 script/smart_langgraph.py에서 import
# =========================================================

from __future__ import annotations
import streamlit as st
import os
import sys
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# =========================================================
# script 폴더를 Python path에 추가
# =========================================================
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# =========================================================
# LangGraph 모듈 import
# =========================================================
from smart_langgraph_for_3_5_2 import (
    # 상수
    YEAR_TO_FILENAME,
    BOT_IDENTITY,
    # RAG Dictionary 함수
    load_rag_dict,
    build_rag_dict_index,
    # 노드 팩토리 및 그래프 빌드
    create_node_functions,
    build_graph,
)

# =========================================================
# Hugging Face 설정
# =========================================================
HF_REPO_ID = "Rosaldowithbaek/smartphoe_overdependence_survey_chromadb"
LOCAL_DB_PATH = "./chroma_db_store"
RAG_DICT_PATH = 'rag_retrieval_dictionary.json'

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="스마트폰 과의존 실태조사 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 커스텀 CSS - 포멀한 디자인
# =========================================================
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* 헤더 스타일 */
    h1 {
        color: #1e3a5f;
        font-weight: 600;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 0.5rem;
    }
    
    /* 가이드 박스 - 포멀한 네이비/그레이 계열 */
    .guide-box {
        background: #1e3a5f;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #0d2137;
    }
    
    .guide-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    
    .guide-item {
        background: rgba(255,255,255,0.1);
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        line-height: 1.6;
        border-left: 3px solid rgba(255,255,255,0.3);
    }
    
    .guide-item:last-child {
        margin-bottom: 0;
    }
    
    .guide-item strong {
        color: #a8c5e2;
    }
    
    .guide-item a {
        color: #7eb8e7;
        text-decoration: underline;
    }
    
    /* 상태 박스 */
    .status-box {
        background-color: #f0f4f8;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #1e3a5f;
        margin: 0.5rem 0;
        font-weight: 500;
        color: #1e3a5f;
    }
    
    /* 검증 상태 배지 */
    .validation-pass {
        color: #1b5e20;
        font-weight: 600;
    }
    
    .validation-fail {
        color: #b71c1c;
        font-weight: 600;
    }
    
    .retry-badge {
        background-color: #fff3e0;
        color: #e65100;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* 다운로드 섹션 */
    .download-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    
    .download-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = None

if "rag_dict" not in st.session_state:
    st.session_state.rag_dict = load_rag_dict(RAG_DICT_PATH)

if "rag_dict_index" not in st.session_state:
    st.session_state.rag_dict_index = build_rag_dict_index(st.session_state.rag_dict)


# =========================================================
# Hugging Face에서 DB 다운로드
# =========================================================
@st.cache_resource
def download_chroma_db():
    """Hugging Face에서 Chroma DB를 다운로드한다."""
    if os.path.exists(LOCAL_DB_PATH) and os.listdir(LOCAL_DB_PATH):
        return LOCAL_DB_PATH, None
    
    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DB_PATH,
            local_dir_use_symlinks=False
        )
        return downloaded_path, None
    except Exception as e:
        return None, str(e)


# =========================================================
# 초기화 함수
# =========================================================
@st.cache_resource
def init_resources():
    """리소스(벡터스토어, LLM)를 초기화한다."""
    # API 키 확인
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        return None, None, "OpenAI API 키가 설정되지 않았습니다."
    
    os.environ['OPENAI_API_KEY'] = api_key
    
    # DB 경로 확인
    if not os.path.exists(LOCAL_DB_PATH):
        return None, None, f"Chroma DB를 찾을 수 없습니다: {LOCAL_DB_PATH}"
    
    try:
        # 임베딩 및 벡터스토어 초기화
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        vectorstore = Chroma(
            persist_directory=LOCAL_DB_PATH,
            embedding_function=embedding,
            collection_name="pdf_pages_with_summary_v2"
        )
        
        # LLM 설정 - 원본과 동일한 모델명 사용
        llms = {
            "router": ChatOpenAI(model="gpt-5-mini", temperature=0),
            "chat_refer": ChatOpenAI(model="gpt-5-mini", temperature=0),
            "parse_year": ChatOpenAI(model="gpt-5-mini", temperature=0),
            "followup": ChatOpenAI(model="gpt-5-mini", temperature=0.2),
            "casual": ChatOpenAI(model="gpt-5-mini", temperature=0.5, max_tokens=500),
            "main": ChatOpenAI(model="gpt-5", temperature=0),
            "rewrite": ChatOpenAI(model="gpt-5-mini", temperature=0),
            "validator": ChatOpenAI(model="gpt-5", temperature=0),
        }
        
        return vectorstore, llms, None
    except Exception as e:
        return None, None, str(e)


# =========================================================
# 테이블 파싱 및 렌더링
# =========================================================
def parse_markdown_table(text: str):
    """마크다운 테이블을 파싱한다."""
    tables = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|') and line.endswith('|'):
            table_lines = []
            start_idx = i
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('|') and line.endswith('|'):
                    table_lines.append(line)
                    i += 1
                elif line.startswith('|---') or line.startswith('| ---'):
                    i += 1
                    continue
                else:
                    break
            
            if len(table_lines) >= 2:
                header_line = table_lines[0]
                headers = [h.strip() for h in header_line.split('|')[1:-1]]
                data_rows = []
                for row_line in table_lines[1:]:
                    if '---' in row_line:
                        continue
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
                
                if headers and data_rows:
                    tables.append({
                        'headers': headers,
                        'rows': data_rows,
                        'start_idx': start_idx,
                        'end_idx': i
                    })
        else:
            i += 1
    return tables


def render_answer_with_tables(answer: str) -> None:
    """테이블이 포함된 답변을 렌더링한다."""
    tables = parse_markdown_table(answer)
    if not tables:
        st.markdown(answer)
        return
    
    lines = answer.split('\n')
    current_pos = 0
    
    for table in tables:
        # 테이블 이전 텍스트 렌더링
        before_text = '\n'.join(lines[current_pos:table['start_idx']])
        if before_text.strip():
            st.markdown(before_text)
        
        # 테이블 렌더링
        try:
            df = pd.DataFrame(table['rows'], columns=table['headers'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except:
            st.markdown("| " + " | ".join(table['headers']) + " |")
            for row in table['rows']:
                st.markdown("| " + " | ".join(row) + " |")
        
        current_pos = table['end_idx']
    
    # 테이블 이후 텍스트 렌더링
    if current_pos < len(lines):
        after_text = '\n'.join(lines[current_pos:])
        if after_text.strip():
            st.markdown(after_text)


# =========================================================
# 상태 업데이트 콜백 생성
# =========================================================
def create_status_callback(status_placeholder):
    """
    Streamlit status_placeholder를 업데이트하는 콜백 함수를 생성한다.
    
    Args:
        status_placeholder: st.empty()로 생성된 placeholder
        
    Returns:
        callable: 상태 텍스트를 받아 placeholder를 업데이트하는 함수
    """
    def callback(status_text: str):
        status_placeholder.markdown(
            f'<div class="status-box">{status_text}</div>', 
            unsafe_allow_html=True
        )
    return callback

# 사용자 가이드 박스 (들여쓰기/개행 때문에 코드블록으로 보이는 문제 방지)
guide_html = """
<div class="guide-box">
  <div class="guide-title">📌 사용 안내</div>

  <div class="guide-item">
    <strong>용도:</strong> 스마트폰 과의존 실태조사 보고서(2020~2024) <strong>정보 검색용</strong>입니다. <br />
    인사이트 제공, 일반 대화, 보고서 외 정보 검색에는 적합하지 않습니다.
  </div>

  <div class="guide-item">
    <strong>검색 팁:</strong> 아래 3가지 요소를 포함하면 정확도가 높아집니다.<br />
    <br />
    <strong>① 연도</strong><br />
    • 2020~2024 중 선택 (미입력 시 2023~2024 적용)<br />
    • 💡 “최근 N년”은 기준연도 계산 후 2020~2024 밖 연도는 제외되어 범위가 좁아질 수 있어 숫자 연도 권장<br />
    <br />
    <strong>② 대상</strong><br />
    • 유아동(만3~9, 보호자응답) / 청소년(10~19) / 성인(20~59) / 60대(60~69, 고령층·시니어)<br />
    • ※ 70대 이상은 조사 대상 아님<br />
    <br />
    <strong>③ 지표</strong><br />
    • 콘텐츠 이용률(%) vs 콘텐츠 이용정도(빈도/점수) 구분해서 입력 등 구체적인 지표명<br />
    <br />
    <strong>TIP</strong><br />
    • 교차조건(성별/대상 등)이나 주제 키워드(숏폼/콘텐츠명/지표명/예방교육 등)를 추가하면 더 정확해집니다.<br />
    • 주제, 대상이 바뀌면 이전 대화 맥락이 오히려 정확한 답변에 방해가 되거나 검색이 안되는 결과로 나타날 수 있습니다. 초기화 이후 재검색하는 것을 권장합니다.<br /> 
    • 과도한 검색결과 방지를 위한 설정으로 인해 일부 연도가 검색 결과에서 누락될 수 있습니다. 그럴 때는 해당 연도를 지정해서 다시 질문해주세요.<br />
    • 보고서 내 유사한 내용이 다수 있어, 검색 성능이 안나올 수 있습니다. 요구하고자하는 바를 확실히 설명해주세요<br />
  </div>

  <div class="guide-item">
    <strong>주의:</strong> AI 답변에 <strong>오류(할루시네이션)</strong>가 있을 수 있습니다. <br />
    검색 결과를 바로 인용하지 마시고, <strong>원문을 통해 확인</strong>한 뒤 정보를 사용하십시오.<br />
    왼쪽의 pdf 보고서 다운로드 혹은
    <a href="https://www.nia.or.kr" target="_blank">NIA 홈페이지</a>에서 원문 확인 권장<br />
    단 pdf 다운로드 클릭시 기존 채팅이 멈출 수 있으니 사전에 혹은 모든 대화가 끝난 이후 다운로드를 해주세요<br />
  </div>
</div>
"""

# =========================================================
# 메인 UI
# =========================================================
def main():
    from textwrap import dedent  # 표준라이브러리라 requirements 추가 불필요

    st.title("📊 스마트폰 과의존 실태조사 분석 시스템")

    # 사이드바
    with st.sidebar:
        st.header("시스템 정보")
        st.markdown(BOT_IDENTITY)

        st.divider()

        # PDF 다운로드 섹션
        st.subheader("📖 보고서 다운로드")
        for year, filename in YEAR_TO_FILENAME.items():
            pdf_path = f"data/{filename}"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"{year}년 보고서",
                        data=f,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

        st.divider()

        # 대화 초기화 버튼
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.clarification_context = None
            st.rerun()

        st.divider()

        # 디버그 모드 토글
        debug_mode = st.checkbox("디버그 모드", value=False)

    # 사용자 가이드 박스 
    st.markdown(guide_html.strip(), unsafe_allow_html=True)
    # DB 다운로드
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다...")
        with st.spinner(f"Hugging Face에서 다운로드 중... ({HF_REPO_ID})"):
            db_path, error = download_chroma_db()

        if error:
            st.error(f"DB 다운로드 실패: {error}")
            return
        else:
            st.success("DB 다운로드 완료!")
            st.rerun()

    # 리소스 초기화
    vectorstore, llms, error = init_resources()

    if error:
        st.error(f"초기화 오류: {error}")
        if "API" in error:
            st.info("Streamlit Secrets에 OPENAI_API_KEY를 설정해주세요.")
            with st.form("api_key_form"):
                api_key = st.text_input("OpenAI API 키", type="password")
                if st.form_submit_button("설정") and api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                    st.rerun()
        return

    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_with_tables(message["content"])
            else:
                st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요... (예: 2024년 청소년 과의존률은?)"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            try:
                # 상태 콜백 생성
                status_callback = create_status_callback(status_placeholder)

                # 노드 함수 생성 (smart_langgraph.py에서 import)
                node_functions = create_node_functions(
                    vectorstore,
                    llms,
                    status_callback,  # 콜백 함수 전달
                    st.session_state.rag_dict_index
                )

                # 그래프 빌드 (smart_langgraph.py에서 import)
                graph = build_graph(node_functions)

                config = {"configurable": {"thread_id": "streamlit_session"}}

                # 그래프 실행
                result = graph.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                        "session_id": "streamlit_session",
                        "clarification_context": st.session_state.clarification_context,
                    },
                    config=config
                )

                # 상태 표시 제거
                status_placeholder.empty()

                # Clarification context 저장
                if result.get("clarification_context"):
                    st.session_state.clarification_context = result["clarification_context"]
                else:
                    st.session_state.clarification_context = None

                # 최종 답변 표시
                final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")

                with answer_placeholder.container():
                    render_answer_with_tables(final_answer)

                # 디버그 정보 표시
                if debug_mode:
                    with st.expander("🔍 디버그 정보", expanded=False):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Intent:** {result.get('intent', 'N/A')}")
                            st.write(f"**Followup:** {result.get('followup_type', 'N/A')}")
                            st.write(f"**Retry Count:** {result.get('retry_count', 0)}")
                            st.write(f"**Default Years Used:** {result.get('used_default_years', False)}")

                            validation_result = result.get('validation_result', 'N/A')
                            if validation_result == "PASS":
                                st.markdown(
                                    f"**Validation:** <span class='validation-pass'>{validation_result}</span>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"**Validation:** <span class='validation-fail'>{validation_result}</span>",
                                    unsafe_allow_html=True
                                )

                        with col2:
                            if result.get("rewritten_queries"):
                                st.write("**Rewritten Queries:**")
                                for q in result["rewritten_queries"][:3]:
                                    st.caption(f"• {q[:50]}...")

                            if result.get("retrieval"):
                                st.write(f"**검색 파일:** {result['retrieval'].get('files_searched', [])}")
                                st.write(f"**문서 수:** {result['retrieval'].get('doc_count', 0)}")

                            if result.get("plan"):
                                st.write(f"**검색 연도:** {result['plan'].get('years', [])}")

                            if result.get("validation_reason"):
                                st.write(f"**Validation Reason:** {result['validation_reason'][:100]}")

                            st.write(f"**Safety:** passed={result.get('safety_passed', 'N/A')}")

                            if result.get("dict_hint"):
                                dh = result["dict_hint"]
                                st.write(f"**Dict Hint - Topic:** {dh.get('topic_code', 'N/A')}")
                                st.write(f"**Dict Hint - Target:** {dh.get('target_group', 'N/A')}")
                                st.write(f"**Anchors:** {dh.get('anchor_terms', [])}")

                # 세션 상태 업데이트
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))

                # 히스토리 길이 제한
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()




















