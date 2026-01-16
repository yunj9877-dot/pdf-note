import os
from PyPDF2 import PdfReader
import streamlit as st

# 최신 패키지 경로 (에러 방지용)
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. 세션 상태 초기화 및 관리 ---
# 사용자가 입력한 키를 저장할 세션 변수 생성
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# 초기화 버튼 클릭 시 실행될 함수
def clear_api_key():
    st.session_state.user_api_key = ""  # 변수 비우기
    # 위젯의 내부 상태도 비우기 위해 입력창의 고유 키 값을 초기화
    if "api_key_input" in st.session_state:
        st.session_state.api_key_input = ""

def process_text(text, api_key): 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main():
    st.set_page_config(page_title="PDF 요약 비서", page_icon="📄")
    
    # --- 2. 사이드바 UI (API 키 관리) ---
    with st.sidebar:
        st.title("🔑 API 설정")
        
        # 입력창: value를 세션 상태와 연결하고, key를 부여하여 직접 제어
        input_key = st.text_input(
            "OpenAI API Key 입력", 
            value=st.session_state.user_api_key,
            type="password", 
            placeholder="sk-...",
            key="api_key_input" # 고유 키 부여
        )
        
        # 키 저장 로직: 입력이 발생하면 세션 변수 업데이트
        if input_key:
            st.session_state.user_api_key = input_key

        col1, col2 = st.columns(2)
        with col1:
            # 초기화 버튼: 클릭 시 clear_api_key 함수 호출
            if st.button("키 초기화", on_click=clear_api_key):
                st.rerun() # 화면 즉시 새로고침하여 공란 반영
        with col2:
            st.link_button("키 발급받기", "https://platform.openai.com/api-keys")

        st.divider()
        st.info("💡 키를 입력하고 엔터를 친 후 PDF를 업로드하세요.")

    # --- 3. 메인 화면 ---
    st.title("🤖 PDF 요약 서비스")
    st.write("문서를 업로드하면 AI가 핵심 내용을 요약해 드립니다.")
    st.divider()

    pdf = st.file_uploader('PDF 파일을 선택하세요', type='pdf', key="pdf_uploader")

    if pdf is not None:
        if not st.session_state.user_api_key:
            st.error("⚠️ 사이드바에서 API 키를 먼저 입력해 주세요!")
            st.stop()

        with st.spinner("PDF 문서 분석 중..."):
            pdf_reader = PdfReader(pdf)
            text = "" 
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted

            try:
                vectorstore = process_text(text, st.session_state.user_api_key)
                
                prompt = ChatPromptTemplate.from_template("""
                당신은 문서 요약 전문가입니다. 아래 내용을 바탕으로 반드시 한국어로 답변하세요.
                내용을 3~5문장으로 명확하게 요약해 주세요.
                
                문서 내용:
                {context}
                """)

                llm = ChatOpenAI(
                    model="gpt-4o-mini", 
                    openai_api_key=st.session_state.user_api_key, 
                    temperature=0
                )
                
                chain = prompt | llm | StrOutputParser()
                
                with st.spinner("AI가 요약 중입니다..."):
                    with get_openai_callback() as cost:
                        docs = vectorstore.similarity_search("문서의 핵심 내용 요약", k=4)
                        context_text = "\n".join([doc.page_content for doc in docs])
                        response = chain.invoke({"context": context_text})

                st.success("✅ 요약 완료!")
                st.subheader("📝 요약 결과")
                st.write(response)
                st.caption(f"💰 비용: ${cost.total_cost:.5f}")
                
            except Exception as e:
                st.error(f"오류 발생: {e}")

if __name__ == '__main__':
    main()