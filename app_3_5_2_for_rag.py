

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Tuple, Optional

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

import final_streamlit_for_github_upload_3_5_2 as core


# =============================================================================
# 경로/상수
# =============================================================================
DATA_DIR = os.getenv("DATA_DIR", "data")            # PDF가 위치한 폴더(요구사항: data)
HF_REPO_ID = core.HF_REPO_ID
LOCAL_DB_PATH = "./chroma_db_store"


YEAR_TO_FILENAME = core.YEAR_TO_FILENAME           # 코어에서 사용하는 매핑과 동일 유지


# =============================================================================
# 포멀 톤: 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="스마트폰 과의존 실태조사 보고서 분석 시스템(2020~2024년)",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 포멀 톤: 커스텀 CSS
# =============================================================================
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.4rem;
        max-width: 1200px;
    }

    html, body, [class*="css"]  {
        font-size: 15px;
    }

    /* 안내 박스(포멀) */
    .guide-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #1e3a8a;
        color: #0f172a;
        padding: 1.1rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }

    .guide-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #0b1f56;
    }

    .guide-item {
        background: #ffffff;
        border: 1px solid #edf2f7;
        padding: 0.75rem 0.95rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #0f172a;
    }

    .guide-item:last-child {
        margin-bottom: 0;
    }

    .status-box {
        background-color: #eff6ff;
        padding: 0.8rem 0.95rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin: 0.5rem 0;
        font-weight: 600;
        color: #0f172a;
    }

    .retry-badge {
        background-color: #fff7ed;
        color: #9a3412;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .validation-pass { color: #166534; font-weight: 700; }
    .validation-fail { color: #b91c1c; font-weight: 700; }

    h1, h2, h3 {
        color: #0b1f56;
    }

    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #0b1f56;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# 포멀 톤: 시스템 소개 문구(BOT_IDENTITY 개선)
# =============================================================================
BOT_IDENTITY = """
본 시스템은 **2020~2024년 「스마트폰 과의존 실태조사」 보고서** 내용을 기반으로,
질의응답 형태의 **단순 정보 탐색**을 지원합니다.

**제공 범위**
- 연도별 스마트폰 과의존 위험군 비율 및 추이
- 대상별(유아동·청소년·성인·60대) 과의존 현황
- 학령별(초·중·고·대학생) 세부 분석
- 과의존 관련 요인(예: SNS, 숏폼, 게임 등) 기술통계/교차 결과
- 조사 설계(표본, 조사방법, 지표 정의) 요약

**유의사항**
- 답변은 검색된 문맥을 요약한 결과이며, 최종 인용 전 원문 확인이 필요합니다.
"""


# =============================================================================
# 세션 상태 초기화
# =============================================================================
def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "clarification_context" not in st.session_state:
        st.session_state.clarification_context = None


# =============================================================================
# Chroma DB 다운로드(Hugging Face)
# =============================================================================
def _find_chroma_dir(snapshot_root: Path) -> Optional[Path]:
    """
    snapshot_root 안에서 Chroma DB 디렉터리를 찾습니다.
    - 일반적으로 'chroma.sqlite3'가 존재하는 디렉터리가 persist_directory입니다.
    """
    sqlite = list(snapshot_root.rglob("chroma.sqlite3"))
    if sqlite:
        return sqlite[0].parent

    # 케이스 대비: sqlite 파일명이 다를 수도 있어 fallback(폴더명에 chroma가 포함된 경우)
    for p in snapshot_root.rglob("*"):
        if p.is_dir() and "chroma" in p.name.lower():
            if list(p.glob("*.sqlite3")) or list(p.glob("*index*")):
                return p
    return None


def download_chroma_db(
    repo_id: str = HF_REPO_ID,
    local_db_path: str = LOCAL_DB_PATH
) -> Tuple[Optional[str], Optional[str]]:
    """
    Hugging Face repo에서 Chroma DB 스냅샷을 다운로드하여 local_db_path로 복사합니다.

    반환:
      - db_path: 성공 시 local_db_path
      - error: 실패 시 오류 메시지
    """
    try:
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception:
            return None, "huggingface_hub 패키지가 필요합니다. (pip install huggingface_hub)"

        snapshot_dir: Optional[Path] = None

        # repo_type이 model/dataset인지 불명이라, 실패 시 순차 폴백함
        for repo_type in [None, "dataset", "model"]:
            try:
                if repo_type is None:
                    try:
                        snapshot_dir = Path(snapshot_download(repo_id=repo_id, local_dir_use_symlinks=False))
                    except TypeError:
                        snapshot_dir = Path(snapshot_download(repo_id=repo_id))
                else:
                    try:
                        snapshot_dir = Path(snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir_use_symlinks=False))
                    except TypeError:
                        snapshot_dir = Path(snapshot_download(repo_id=repo_id, repo_type=repo_type))
                break
            except Exception:
                snapshot_dir = None

        if snapshot_dir is None:
            return None, f"Hugging Face 다운로드 실패: {repo_id}"

        chroma_src = _find_chroma_dir(snapshot_dir)
        if chroma_src is None:
            return None, f"Hugging Face 스냅샷에서 Chroma DB 디렉터리를 찾지 못했습니다: {snapshot_dir}"

        local_path = Path(local_db_path)
        if local_path.exists():
            shutil.rmtree(local_path)

        shutil.copytree(chroma_src, local_path)
        return str(local_path), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# =============================================================================
# 리소스 초기화(Streamlit 캐시)
# =============================================================================
@st.cache_resource(show_spinner=False)
def _init_resources_cached(openai_api_key: str, persist_dir: str) -> Tuple[Any, Any, Optional[str]]:
    """
    Streamlit rerun 시 중복 초기화를 막기 위한 캐시 wrapper입니다.
    """
    return core.init_resources(openai_api_key=openai_api_key, persist_dir=persist_dir)


# =============================================================================
# 출력 렌더(표 포함)
# =============================================================================
def render_answer_with_tables(answer_markdown: str) -> None:
    st.markdown(answer_markdown)


# =============================================================================
# 메인(요구사항 4 흐름 유지)
# =============================================================================
def main() -> None:
    _init_session_state()

    st.title("📊 스마트폰 과의존 실태조사 분석 시스템(2020~2024)")

    # =========================
    # 사이드바
    # =========================
    with st.sidebar:
        st.header("📋 시스템 정보")
        st.markdown(BOT_IDENTITY)

        st.divider()

        # --------- PDF 다운로드(요구사항 3) ---------
        st.subheader("📥 보고서 PDF 다운로드")
        year = st.selectbox(
            "연도 선택",
            options=sorted(YEAR_TO_FILENAME.keys()),
            index=len(YEAR_TO_FILENAME) - 1,
            format_func=lambda y: f"{y}년",
        )
        pdf_name = YEAR_TO_FILENAME.get(year)
        pdf_path = Path(DATA_DIR) / (pdf_name or "")

        if pdf_name and pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "선택 연도 PDF 다운로드",
                    data=f,
                    file_name=pdf_name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.caption(f"경로: {pdf_path}")
        else:
            st.caption("⚠️ 해당 PDF 파일이 data 폴더에 없습니다.")

        st.divider()

        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.clarification_context = None
            st.rerun()

        st.divider()

        debug_mode = st.checkbox("🔧 디버그 모드", value=False)

    # =========================
    # 사용자 가이드 박스
    # =========================
    st.markdown(
        """
<div class="guide-box">
  <div class="guide-title">📌 사용 안내</div>

  <div class="guide-item">
    <strong>ℹ️ 용도</strong><br/>
    스마트폰 과의존 실태조사 보고서(2020~2024) 기반 <strong>단순 정보 검색</strong> 용도입니다.<br/>
    인사이트 제공, 일반 대화, 보고서 외 정보 검색에는 적합하지 않습니다.
  </div>

  <div class="guide-item">
    <strong>💡 검색 팁</strong><br/>
    질문은 <strong>가능한 한 구체적으로</strong> 작성해 주십시오.<br/>
    예) "과의존률" → "2024년 청소년 스마트폰 과의존 위험군 비율"<br/>
    <strong>과도한 검색결과 방지를 위한 설정으로 인해 일부 연도가 검색 결과에서 누락될 수 있습니다. 그럴 때는 해당 연도를 지정해서 다시 질문해주세요.</strong><br/>
    <strong>보고서 내 유사한 내용이 다수 있어, 검색 성능이 안나올 수 있습니다. 요구하고자하는 바를 확실히 설명해주세요</strong><br/>
    예) "숏폼과 과의존" → "숏폼 이용률에 따른 과의존 차이" or "과의존위험군별 숏폼 이용 특성의 차이" <br/>
    <strong>연속된 요청은 성능을 저하하는 원인이 될 수 있습니다. 왼쪽 대화초기화를 통해 새롭게 세션을 구성하세요.</strong><br/>
    <strong>확장질문을 할때에도 최대한 질문을 구체적으로 작성해야 검색에 도움이 됩니다.</strong><br/>
    예) "과의존 여부에 따른 sns 이용정도" → "AI 답변" → "청소년은?" x "과의존 여부에 따른 sns이용정도에 대해 청소년은 어떻게 나타나?"<br/>
  </div>

  <div class="guide-item">
    <strong>⚠️ 유의</strong><br/>
    AI 답변에는 오류(할루시네이션)가 포함될 수 있습니다.<br/>
    <strong>검색 결과를 즉시 인용하지 마시고</strong>, 원문을 확인한 뒤 최종 활용해 주십시오.</br>
    📥 보고서 PDF 다운로드를 클릭하면 원문을 확인할 수 있습니다. (페이지 번호는 본문 페이지 번호가 아닌 pdf 파일의 페이지 순서를 의미합니다.) 
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # =========================
    # DB 다운로드(요구사항 1)
    # =========================
    local_dir = Path(LOCAL_DB_PATH)
    if (not local_dir.exists()) or (local_dir.exists() and not any(local_dir.iterdir())):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다...")
        with st.spinner(f"Hugging Face에서 다운로드 중... ({HF_REPO_ID})"):
            db_path, error = download_chroma_db()

        if error:
            st.error(f"DB 다운로드 실패: {error}")
            return

        st.success("DB 다운로드 완료!")
        st.rerun()

    # =========================
    # 리소스 초기화
    # =========================
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("초기화 오류: OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.info("Streamlit Secrets에 OPENAI_API_KEY를 설정하거나, 아래 입력창에 API 키를 입력해 주십시오.")
        with st.form("api_key_form"):
            user_key = st.text_input("OpenAI API 키", type="password")
            if st.form_submit_button("설정") and user_key:
                os.environ["OPENAI_API_KEY"] = user_key.strip()
                st.rerun()
        return

    vectorstore, llms, error = _init_resources_cached(api_key, LOCAL_DB_PATH)
    if error:
        st.error(f"초기화 오류: {error}")
        return

    # =========================
    # 채팅 히스토리 표시
    # =========================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_with_tables(message["content"])
            else:
                st.markdown(message["content"])

    # =========================
    # 사용자 입력
    # =========================
    if prompt := st.chat_input("질문을 입력하세요... (예: 2024년 청소년 과의존률은?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            try:
                # (요구사항 4) 샘플 흐름 유지
                node_functions = core.create_node_functions(vectorstore, llms, status_placeholder)
                graph = core.build_graph(node_functions)

                config = {
                    "configurable": {"thread_id": "streamlit_session"},
                    "recursion_limit": 80,
                }

                result = graph.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                        "session_id": "streamlit_session",
                        "clarification_context": st.session_state.clarification_context,
                    },
                    config=config,
                )

                status_placeholder.empty()

                # Clarification context 저장
                if result.get("clarification_context"):
                    st.session_state.clarification_context = result["clarification_context"]
                else:
                    st.session_state.clarification_context = None

                final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")

                with answer_placeholder.container():
                    render_answer_with_tables(final_answer)

                # 디버그 정보
                if debug_mode:
                    with st.expander("🔍 디버그 정보 (v5)", expanded=False):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Intent:** {result.get('intent', 'N/A')}")
                            st.write(f"**Followup:** {result.get('followup_type', 'N/A')}")
                            st.write(f"**Retry Count:** {result.get('retry_count', 0)}")
                            st.write(f"**Default Years Used:** {result.get('used_default_years', False)}")

                            validation_result = result.get("validation_result", "N/A")
                            if validation_result == "PASS":
                                st.markdown(
                                    f"**Validation:** <span class='validation-pass'>{validation_result}</span>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"**Validation:** <span class='validation-fail'>{validation_result}</span>",
                                    unsafe_allow_html=True,
                                )

                        with col2:
                            if result.get("rewritten_queries"):
                                st.write("**Rewritten Queries:**")
                                for q in result["rewritten_queries"][:3]:
                                    st.caption(f"• {q[:70]}...")

                        if result.get("retrieval"):
                            st.write(f"**검색 파일:** {result['retrieval'].get('files_searched', [])}")
                            st.write(f"**문서 수:** {result['retrieval'].get('doc_count', 0)}")

                        if result.get("plan"):
                            st.write(f"**검색 연도:** {result['plan'].get('years', [])}")

                        if result.get("validation_reason"):
                            st.write(f"**Validation Reason:** {str(result['validation_reason'])[:180]}")

                        st.write(f"**Safety:** passed={result.get('safety_passed', 'N/A')}")

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))

                # 히스토리 길이 제한(원 예시 유지)
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {type(e).__name__}: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

