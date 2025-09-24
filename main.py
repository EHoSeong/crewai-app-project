import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from crewai_tools import SerperDevTool

# .env 파일에서 환경 변수 로드
load_dotenv()

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# LLM 객체와 툴은 앱 시작 시점에 한 번만 정의합니다.
gemini_llm = LLM(model="gemini/gemini-1.5-flash", verbose=True, temperature=0.5)
search_tool = SerperDevTool()

# CORS 미들웨어 설정
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 사용자 입력 모델 정의
class UserInput(BaseModel):
    topic: str


# 사용자 질문 유형을 분류하는 도우미 함수
def classify_job_type(topic: str) -> str:
    """주제가 '직업'인지 '자영업/프리랜서'인지 분류"""
    entrepreneur_keywords = [
        "사장",
        "창업",
        "사업",
        "가게",
        "자영업",
        "프리랜서",
        "컨설턴트",
        "운영",
        "ceo",
    ]
    if any(keyword in topic for keyword in entrepreneur_keywords):
        return "자영업"
    else:
        return "직업"


# 루트 엔드포인트: index.html 파일 제공
@app.get("/")
def read_root():
    return FileResponse("index.html")


# CrewAI 실행 엔드포인트
@app.post("/run-crew")
async def run_crew(user_input: UserInput):
    """
    사용자의 입력 주제에 따라 CrewAI를 동적으로 구성하고 실행
    """
    try:
        topic = user_input.topic
        job_type = classify_job_type(topic)

        # job_type에 따라 Agent의 목표(goal)와 Task의 설명(description) 동적 변경
        if job_type == "자영업":
            # 자영업 관련 프롬프트
            researcher_goal = (
                f"최신 {topic} 시장 트렌드와 예상 순이익 구조, 성공 전략을 파악하는 것"
            )
            researcher_backstory = f"다양한 시장 데이터와 경쟁사 분석을 통해 {topic} 사업의 기회를 발굴하는 전문가입니다. 정량적, 정성적 데이터를 모두 활용하여 신뢰성 높은 인사이트를 제공합니다."
            research_task_desc = f"{topic} 창업에 필요한 자본금, 월별 예상 수익 및 지출 구조, 성공을 위한 마케팅 전략을 조사하고 분석하세요."
            research_expected_output = f"상세한 {topic} 시장 트렌드, 필요 역량, 예상 순이익 구조, 사업 전략 가이드 보고서."

            writer_goal = f"심층 분석된 {topic} 시장 데이터를 바탕으로, 창업자도 쉽게 이해하고 즉시 활용할 수 있는 실용적인 가이드 콘텐츠를 작성하세요."
            writer_backstory = f"복잡한 시장 데이터를 일반인이 이해하기 쉬운 명확하고 흥미로운 이야기로 재구성하는 데 탁월한 능력을 가졌습니다. 창업자들이 성공적인 사업을 설계하는 데 실질적인 도움을 주는 글을 씁니다."
            writing_task_desc = f'시장 분석가(researcher)의 보고서를 참고하여, "{topic} 성공을 위한 완벽 가이드"를 작성하세요. 보고서는 시장 동향, 창업 비용, 예상 수익, 사업 성공 체크리스트를 포함해야 합니다.'
            writing_expected_output = f"창업자들이 {topic} 사업을 시작하기 위해 필요한 모든 정보를 담은 실용적인 가이드 콘텐츠.\n- 매력적인 제목\n- 사업 소개\n- 핵심 트렌드 요약\n- 창업 비용 및 예상 수익 구조\n- 사업 성공 체크리스트"

            fact_checker_goal = f"{topic} 성공 가이드 콘텐츠의 모든 데이터와 사실 관계를 철저히 검증하여, 정확성을 보장하는 것."
            fact_checker_backstory = f"최신 통계와 신뢰할 수 있는 소스를 바탕으로 모든 정보를 교차 검증하는 전문가입니다. 데이터의 오류나 왜곡을 찾아내고, 신뢰할 수 있는 답변을 만드는 데 기여합니다."
            fact_checking_task_desc = f"작가(writer)의 초안을 검토하고, 모든 수치, 시장 트렌드, 전략의 사실 관계를 재검증하세요. 불확실하거나 오래된 정보는 수정하고, 최종 보고서가 100% 정확하도록 만드세요."
            fact_checking_expected_output = f'모든 사실이 검증되고 오류가 수정된 최종 버전의 "{topic} 성공 가이드" 보고서.'

        else:  # 직업 관련 프롬프트
            researcher_goal = f"2025년 최신 {topic} 직업 시장의 핵심 트렌드, 요구되는 기술 스택, 지역 및 경력별 최신 평균 연봉 데이터, 그리고 성공적인 취업을 위한 포트폴리오 전략을 구체적인 수치와 사례를 포함하여 심층적으로 분석하세요."
            researcher_backstory = f"다양한 시장 보고서, 채용 공고 데이터, 기술 동향 분석 자료를 종합하여 {topic} 직업 시장의 미래 기회를 발굴하는 전문가입니다. 정량적, 정성적 데이터를 모두 활용하여 신뢰성 높은 인사이트를 제공합니다."
            research_task_desc = f"2025년 현재, {topic} 직업의 주요 트렌드는 무엇인가요? 이 직업을 얻기 위해 필수적인 기술 스택 5가지와, 신입/경력자별 평균 연봉 범위를 조사하세요. 또한, 취업 시 면접관에게 깊은 인상을 남길 수 있는 포트폴리오 구성 팁을 3가지 이상 포함하여 상세 보고서를 작성하세요."
            research_expected_output = f"다음 내용을 포함하는 상세 분석 보고서:\n- 2025년 {topic} 직업 시장 트렌드\n- 필수 기술 스택 (최소 5개)\n- 신입/경력별 평균 연봉 (범위로 제시)\n- 포트폴리오 구성 팁 (최소 3가지)\n- 모든 정보는 최신 데이터를 기반으로 해야 합니다."

            writer_goal = f"심층 분석된 {topic} 직업 시장 데이터를 바탕으로, 초보자도 쉽게 이해하고 즉시 활용할 수 있는 실용적인 가이드 콘텐츠를 작성하세요."
            writer_backstory = f"복잡한 시장 데이터를 일반인이 이해하기 쉬운 명확하고 흥미로운 이야기로 재구성하는 데 탁월한 능력을 가졌습니다. 구직자들이 커리어 로드맵을 설계하는 데 실질적인 도움을 주는 글을 씁니다."
            writing_task_desc = f'시장 트렌드 분석가(researcher)의 보고서를 참고하여, "{topic} 전문가를 위한 2025년 완벽 가이드"라는 제목의 최종 콘텐츠를 제작하세요. 콘텐츠는 직업의 매력적인 소개글로 시작하고, 분석된 데이터(트렌드, 기술, 연봉)를 보기 좋게 정리한 후, 마지막에는 실용적인 "액션 플랜 체크리스트"를 추가하여 독자가 바로 행동할 수 있도록 유도하세요.'
            writing_expected_output = f"구직자들이 {topic} 전문가가 되기 위해 필요한 모든 정보를 담은 실용적인 가이드 콘텐츠.\n- 매력적인 제목\n- 직업 소개\n- 핵심 트렌드 요약\n- 필요 역량 및 기술 스택\n- 예상 연봉\n- 최종 액션 플랜 체크리스트"

            # FactChecker 에이전트와 태스크 정의
            fact_checker_goal = f'"{topic} 전문가를 위한 2025년 완벽 가이드" 콘텐츠의 모든 데이터와 사실 관계를 철저히 검증하여, 정확성을 보장하는 것.'
            fact_checker_backstory = f"최신 통계와 신뢰할 수 있는 소스를 바탕으로 모든 정보를 교차 검증하는 전문가입니다. 데이터의 오류나 왜곡을 찾아내고, 신뢰할 수 있는 답변을 만드는 데 기여합니다."
            fact_checking_task_desc = f'작가(writer)가 작성한 "{topic} 전문가를 위한 2025년 완벽 가이드" 콘텐츠를 최종적으로 검토하고, 모든 수치와 정보의 사실 관계를 재검증하세요. 불확실하거나 오래된 정보는 수정하고, 최종 보고서가 100% 정확하도록 만드세요.'
            fact_checking_expected_output = f'모든 사실이 검증되고 오류가 수정된 최종 버전의 "{topic} 전문가를 위한 2025년 완벽 가이드" 콘텐츠.'
        # Agent와 Task 인스턴스 생성
        researcher = Agent(
            role=researcher_goal,
            goal=researcher_goal,
            backstory=researcher_backstory,
            llm=gemini_llm,
            tools=[search_tool],
            allow_delegation=True,
        )
        writer = Agent(
            role=writer_goal,
            goal=writer_goal,
            backstory=writer_backstory,
            llm=gemini_llm,
            allow_delegation=False,
        )
        fact_checker = Agent(
            role=fact_checker_goal,
            goal=fact_checker_goal,
            backstory=fact_checker_backstory,
            llm=gemini_llm,
            tools=[search_tool],  # FactChecker도 웹 검색 도구가 필요할 수 있습니다.
            allow_delegation=False,
        )

        research_task = Task(
            description=research_task_desc,
            expected_output=research_expected_output,
            agent=researcher,
            tools=[search_tool],
        )
        writing_task = Task(
            description=writing_task_desc,
            expected_output=writing_expected_output,
            agent=writer,
        )
        fact_checking_task = Task(
            description=fact_checking_task_desc,
            expected_output=fact_checking_expected_output,
            agent=fact_checker,
            context=[
                writing_task,
                research_task,
            ],  # writer의 초안과 researcher의 원본 데이터를 모두 검토
        )

        # Crew 구성 및 실행
        project_crew = Crew(
            agents=[researcher, writer, fact_checker],
            tasks=[research_task, writing_task, fact_checking_task],
            process=Process.sequential,
            verbose=False,
        )

        result = project_crew.kickoff()
        final_output = result if isinstance(result, str) else result.raw

        print("######## 분석끝 결과나왔음")
        return {"result": final_output}

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
