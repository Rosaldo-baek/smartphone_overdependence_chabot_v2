# 스마트폰 과의존 실태조사 RAG 챗봇

2020~2024년 스마트폰 과의존 실태조사 보고서 분석 시스템입니다.

## 주요 기능

- **연도별 분석**: 2020~2024년 5개년 보고서 데이터 검색
- **멀티연도 비교**: 여러 연도 데이터를 표 형식으로 비교
- **대화 맥락 유지**: 후속 질문 지원
- **실시간 진행 상태**: 분석 → 검색 → 생성 → 검증 단계 표시

pip install -r requirements.txt

## 질문 예시

- "2024년 청소년 과의존률은?"
- "2021년부터 2024년까지 학령별 과의존률 변화"
- "숏폼 이용과 과의존의 관계는?"
- "가구원 수에 따른 과의존률 비교"

## 기술 스택

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-5o / GPT-5o-mini
- **Embedding**: text-embedding-3-large
- **Vector DB**: Chroma
- **Orchestration**: LangGraph
