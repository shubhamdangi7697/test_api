from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from typing import List, Dict, Optional, Literal
import os
import uuid
from datetime import datetime
import random
import asyncio
import json
from google import genai
from google.genai import types

app = FastAPI(title="AWS DVA-C02 Certification Practice API with Gemini", version="2.0")

# MongoDB Setup
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb+srv://shubhamdmca:OdS48xyWAmxO3Q0H@cluster0.ng5gagk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"))
db = mongo_client.aws_dva_c02_practice

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("aoi_key","AIzaSyDtAs2OmoBfqexwZxY2hhywwBkq8ktWhso"))

# Data Models
class DVAQuestion(BaseModel):
    question_id: str
    domain: Literal["development", "security", "deployment", "troubleshooting"]
    task_number: int
    question_type: Literal["multiple_choice", "multiple_response"]
    question: str
    options: List[str]
    correct_answers: List[str]
    explanation: str
    difficulty: Literal["easy", "medium", "hard"]
    aws_services: List[str]
    is_scored: bool = True
    scenario_based: bool = False

class DVAPracticeSet(BaseModel):
    set_id: str
    set_number: int
    topic: str = "AWS Certified Developer Associate (DVA-C02)"
    questions: List[DVAQuestion]
    created_at: datetime
    total_questions: int = 65
    scored_questions: int = 50
    unscored_questions: int = 15
    domain_distribution: Dict[str, int]

class UserSession(BaseModel):
    session_id: str
    user_id: str
    set_id: str
    started_at: datetime
    time_limit: int = 7800  # 130 minutes in seconds
    current_question_index: int = 0
    is_completed: bool = False

# AWS DVA-C02 Content Configuration
DVA_C02_DOMAINS = {
    "development": {
        "weight": 0.32,
        "questions_per_set": 21,  # 32% of 65
        "tasks": {
            1: "Develop code for applications hosted on AWS",
            2: "Develop code for AWS Lambda", 
            3: "Use data stores in application development"
        },
        "services": ["Lambda", "API Gateway", "DynamoDB", "S3", "SQS", "SNS", "Kinesis", "Step Functions"],
        "concepts": ["Event-driven architecture", "Microservices", "Serverless", "APIs", "SDKs"]
    },
    "security": {
        "weight": 0.26,
        "questions_per_set": 17,  # 26% of 65
        "tasks": {
            1: "Implement authentication and/or authorization for applications and AWS services",
            2: "Implement encryption by using AWS services",
            3: "Manage sensitive data in application code"
        },
        "services": ["IAM", "Cognito", "KMS", "Secrets Manager", "STS", "Certificate Manager"],
        "concepts": ["Least privilege", "RBAC", "Encryption", "JWT", "OAuth", "SAML"]
    },
    "deployment": {
        "weight": 0.24,
        "questions_per_set": 16,  # 24% of 65
        "tasks": {
            1: "Prepare application artifacts to be deployed to AWS",
            2: "Test applications in development environments",
            3: "Automate deployment testing",
            4: "Deploy code by using AWS CI/CD services"
        },
        "services": ["CodePipeline", "CodeBuild", "CodeDeploy", "CloudFormation", "SAM", "CDK"],
        "concepts": ["CI/CD", "Blue/green deployment", "Canary deployment", "IaC", "Testing"]
    },
    "troubleshooting": {
        "weight": 0.18,
        "questions_per_set": 11,  # 18% of 65
        "tasks": {
            1: "Assist in a root cause analysis",
            2: "Instrument code for observability",
            3: "Optimize applications by using AWS services and features"
        },
        "services": ["CloudWatch", "X-Ray", "CloudTrail", "ElastiCache"],
        "concepts": ["Monitoring", "Logging", "Tracing", "Performance optimization", "Debugging"]
    }
}

# Enhanced Question Generator with Google GenAI
class DVAQuestionGenerator:
    def __init__(self):
        self.client = gemini_client
        
    async def generate_domain_task_questions(
        self, domain: str, task_num: int, task_description: str, 
        question_count: int, set_number: int
    ) -> List[DVAQuestion]:
        """Generate questions for a specific domain task using Gemini"""
        domain_config = DVA_C02_DOMAINS[domain]
        
        prompt = self._create_detailed_prompt(
            domain, task_description, question_count, set_number, domain_config
        )
        
        try:
            # Use Gemini GenAI client to generate content
            response = await self._call_gemini_api(prompt)
            questions = self._parse_gemini_response(response, domain, task_num)
            return questions
            
        except Exception as e:
            print(f"Error generating questions for {domain}: {str(e)}")
            return []
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Make API call to Gemini using the genai client"""
        try:
            # Configure generation parameters
            config = types.GenerateContentConfig(
                system_instruction="You are an expert AWS Certified Developer Associate exam creator. Generate realistic, practical exam questions that match the official DVA-C02 format.",
                max_output_tokens=8192,
                temperature=0.7,
                top_p=0.8
            )
            
            # Generate content using Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def _create_detailed_prompt(
        self, domain: str, task_description: str, question_count: int, 
        set_number: int, domain_config: Dict
    ) -> str:
        """Create comprehensive prompt for Gemini"""
        return f"""
        Generate {question_count} AWS Certified Developer Associate (DVA-C02) exam questions for Practice Set #{set_number}.
        
        CRITICAL REQUIREMENTS:
        - Domain: {domain.title()} ({domain_config['weight']*100}% of exam)
        - Task: {task_description}
        - Questions must be UNIQUE and not duplicate any previous sets
        - Follow exact DVA-C02 exam format and difficulty
        - Include both multiple-choice (4 options) and multiple-response (5+ options) questions
        - Mix scenario-based and knowledge-based questions
        - Ensure practical, hands-on focus matching real exam
        
        Focus Areas:
        - AWS Services: {', '.join(domain_config['services'])}
        - Key Concepts: {', '.join(domain_config['concepts'])}
        - Set Uniqueness Factor: Practice Set #{set_number} - ensure questions are distinct
        
        Question Requirements:
        - Scenario-based questions with realistic use cases
        - Code snippets where appropriate (Python, Node.js, Java)
        - AWS CLI commands and SDK usage examples
        - Troubleshooting scenarios with logs and error messages
        - Best practices and anti-patterns
        - Security considerations and IAM policies
        - Performance optimization techniques
        
        Output Format (Valid JSON only):
        {{
            "questions": [
                {{
                    "question_type": "multiple_choice",
                    "question": "A developer is building a serverless application...",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answers": ["Option B"],
                    "explanation": "Detailed explanation with AWS documentation references...",
                    "difficulty": "medium",
                    "aws_services": ["Lambda", "API Gateway"],
                    "scenario_based": true
                }}
            ]
        }}
        """
    
    def _parse_gemini_response(self, response: str, domain: str, task_num: int) -> List[DVAQuestion]:
        """Parse Gemini response into DVAQuestion objects"""
        questions = []
        
        try:
            # Clean response and extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            data = json.loads(json_str)
            
            for q_data in data.get('questions', []):
                question = DVAQuestion(
                    question_id=str(uuid.uuid4()),
                    domain=domain,
                    task_number=task_num,
                    question_type=q_data.get('question_type', 'multiple_choice'),
                    question=q_data.get('question', ''),
                    options=q_data.get('options', []),
                    correct_answers=q_data.get('correct_answers', []),
                    explanation=q_data.get('explanation', ''),
                    difficulty=q_data.get('difficulty', 'medium'),
                    aws_services=q_data.get('aws_services', []),
                    scenario_based=q_data.get('scenario_based', False)
                )
                questions.append(question)
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            print(f"Response: {response[:500]}...")
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            
        return questions

# Helper Functions
def calculate_time_remaining(session: dict) -> int:
    """Calculate remaining time in seconds"""
    elapsed = (datetime.now() - session["started_at"]).total_seconds()
    return max(0, session["time_limit"] - int(elapsed))

def is_session_expired(session: dict) -> bool:
    """Check if exam session has expired"""
    return calculate_time_remaining(session) <= 0

async def generate_single_set_with_gemini(generator: DVAQuestionGenerator, set_number: int):
    """Generate a single practice set with Gemini"""
    all_questions = []
    domain_distribution = {}
    
    # Generate questions for each domain using Gemini
    for domain, config in DVA_C02_DOMAINS.items():
        questions_needed = config["questions_per_set"]
        domain_distribution[domain] = questions_needed
        
        # Distribute questions across tasks within the domain
        questions_per_task = questions_needed // len(config["tasks"])
        remaining_questions = questions_needed % len(config["tasks"])
        
        for task_num, task_description in config["tasks"].items():
            task_questions = questions_per_task
            if remaining_questions > 0:
                task_questions += 1
                remaining_questions -= 1
            
            # Generate questions using Gemini for this specific task
            task_questions_list = await generator.generate_domain_task_questions(
                domain, task_num, task_description, task_questions, set_number
            )
            all_questions.extend(task_questions_list)
    
    # Randomly mark 15 questions as unscored
    scored_indices = random.sample(range(len(all_questions)), min(50, len(all_questions)))
    for i, question in enumerate(all_questions):
        question.is_scored = i in scored_indices
    
    # Shuffle questions
    random.shuffle(all_questions)
    
    return DVAPracticeSet(
        set_id=str(uuid.uuid4()),
        set_number=set_number,
        questions=all_questions,
        created_at=datetime.now(),
        domain_distribution=domain_distribution
    )




@app.get("/dva-questions-by-set")
async def get_all_questions_by_set_number(
    set_number: int,
    include_answers: bool = False,
    user_id: Optional[str] = None
):
    """Get all questions from a specific practice set by set number"""
    try:
        # Validate set number range
        if set_number < 1 or set_number > 25:
            raise HTTPException(
                status_code=400, 
                detail="Set number must be between 1 and 25"
            )
        
        # Find practice set by set number
        practice_set = db.dva_practice_sets.find_one({"set_number": set_number})
        if not practice_set:
            raise HTTPException(
                status_code=404, 
                detail=f"Practice set {set_number} not found"
            )
        
        # Get user's previous responses if user_id provided
        user_responses = {}
        if user_id:
            responses = list(db.dva_responses.find({
                "user_id": user_id,
                "set_id": practice_set["set_id"]
            }))
            user_responses = {
                r["question_id"]: {
                    "selected_answers": r.get("selected_answers", []),
                    "is_correct": r.get("is_correct", False),
                    "skipped": r.get("skipped", False)
                }
                for r in responses
            }
        
        # Format questions based on include_answers parameter
        formatted_questions = []
        for i, question in enumerate(practice_set["questions"]):
            question_data = {
                "question_number": i + 1,
                "question_id": question["question_id"],
                "domain": question["domain"],
                "task_number": question["task_number"],
                "question_type": question["question_type"],
                "question": question["question"],
                "options": question["options"],
                "difficulty": question["difficulty"],
                "aws_services": question["aws_services"],
                "is_scored": question["is_scored"],
                "scenario_based": question.get("scenario_based", False)
            }
            
            # Include answers and explanations if requested
            if include_answers:
                question_data.update({
                    "correct_answers": question["correct_answers"],
                    "explanation": question["explanation"]
                })
            
            # Include user's previous response if available
            if user_id and question["question_id"] in user_responses:
                question_data["user_response"] = user_responses[question["question_id"]]
            
            formatted_questions.append(question_data)
        
        # Calculate domain distribution
        domain_counts = {}
        for question in practice_set["questions"]:
            domain = question["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "set_info": {
                "set_id": practice_set["set_id"],
                "set_number": set_number,
                "total_questions": len(practice_set["questions"]),
                "scored_questions": len([q for q in practice_set["questions"] if q["is_scored"]]),
                "unscored_questions": len([q for q in practice_set["questions"] if not q["is_scored"]]),
                "created_at": practice_set["created_at"]
            },
            "domain_distribution": domain_counts,
            "questions": formatted_questions,
            "exam_format": "AWS DVA-C02",
            "time_limit_minutes": 130,
            "passing_score": 720,
            "includes_answers": include_answers,
            "user_progress": {
                "user_id": user_id,
                "attempted_questions": len(user_responses) if user_id else 0,
                "completion_percentage": round((len(user_responses) / 65) * 100, 1) if user_id else 0
            } if user_id else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








# API Endpoints
@app.post("/generate-dva-c02-sets")
async def generate_complete_dva_sets(background_tasks: BackgroundTasks):
    """Generate all 25 AWS DVA-C02 practice sets using Gemini"""
    try:
        # Check if sets already exist
        existing_count = db.dva_practice_sets.count_documents({})
        if existing_count >= 25:
            return {
                "message": "25 DVA-C02 practice sets already exist",
                "existing_sets": existing_count,
                "use_endpoint": "/list-dva-sets"
            }
        
        # Generate sets using Gemini in background
        generator = DVAQuestionGenerator()
        background_tasks.add_task(generate_sets_with_gemini, generator, 25)
        
        return {
            "message": "Generating 25 unique AWS DVA-C02 practice sets using Gemini LLM",
            "total_questions": 25 * 65,
            "gemini_model": "gemini-2.0-flash-exp",
            "estimated_completion": "15-20 minutes",
            "check_status": "/generation-status"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_sets_with_gemini(generator: DVAQuestionGenerator, total_sets: int):
    """Background task to generate practice sets using Gemini"""
    for set_number in range(1, total_sets + 1):
        print(f"Generating practice set {set_number}/25 using Gemini...")
        
        try:
            practice_set = await generate_single_set_with_gemini(generator, set_number)
            
            # Store in MongoDB
            db.dva_practice_sets.insert_one(practice_set.dict())
            print(f"Practice set {set_number} completed and stored")
            
        except Exception as e:
            print(f"Error generating set {set_number}: {str(e)}")
            continue






# Add these endpoints to the existing FastAPI implementation

@app.get("/list-dva-sets")
async def list_available_sets():
    """List all available DVA-C02 practice sets"""
    try:
        sets = list(db.dva_practice_sets.find(
            {}, 
            {"set_id": 1, "set_number": 1, "created_at": 1, "total_questions": 1}
        ).sort("set_number", 1))
        
        return {
            "total_sets": len(sets),
            "sets": [
                {
                    "set_id": s["set_id"],
                    "set_number": s["set_number"], 
                    "total_questions": s["total_questions"],
                    "created_at": s["created_at"]
                }
                for s in sets
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-dva-exam")
async def start_dva_exam_session(user_id: str, set_id: str):
    """Start a timed DVA-C02 exam session"""
    try:
        # Verify practice set exists
        practice_set = db.dva_practice_sets.find_one({"set_id": set_id})
        if not practice_set:
            raise HTTPException(status_code=404, detail="Practice set not found")
        
        # Check for existing active session
        existing_session = db.user_sessions.find_one({
            "user_id": user_id,
            "set_id": set_id,
            "is_completed": False
        })
        
        if existing_session:
            return {
                "message": "Resume existing session",
                "session_id": existing_session["session_id"],
                "time_remaining": calculate_time_remaining(existing_session),
                "current_question": existing_session["current_question_index"],
                "total_questions": 65
            }
        
        # Create new session
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            set_id=set_id,
            started_at=datetime.now()
        )
        
        db.user_sessions.insert_one(session.dict())
        
        return {
            "session_id": session.session_id,
            "time_limit_minutes": 130,
            "total_questions": 65,
            "exam_format": "DVA-C02",
            "instructions": "This is a timed practice exam. You have 130 minutes to complete 65 questions.",
            "set_number": practice_set["set_number"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dva-exam-question")
async def get_dva_exam_question(session_id: str):
    """Get current question for DVA-C02 exam session"""
    try:
        # Get session
        session = db.user_sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if time expired
        if is_session_expired(session):
            await end_session_due_to_timeout(session_id)
            raise HTTPException(status_code=408, detail="Exam time expired")
        
        # Get practice set
        practice_set = db.dva_practice_sets.find_one({"set_id": session["set_id"]})
        
        # Get current question
        current_index = session["current_question_index"]
        if current_index >= len(practice_set["questions"]):
            # Mark session as completed
            db.user_sessions.update_one(
                {"session_id": session_id},
                {"$set": {"is_completed": True}}
            )
            return {"message": "Exam completed", "completed": True}
        
        question = practice_set["questions"][current_index]
        
        return {
            "session_id": session_id,
            "question_number": current_index + 1,
            "total_questions": 65,
            "question_id": question["question_id"],
            "domain": question["domain"],
            "question_type": question["question_type"],
            "question": question["question"],
            "options": question["options"],
            "difficulty": question["difficulty"],
            "aws_services": question["aws_services"],
            "time_remaining": calculate_time_remaining(session),
            "is_scenario_based": question.get("scenario_based", False),
            "progress": {
                "completed": current_index,
                "remaining": 65 - current_index
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-dva-answer")
async def submit_dva_answer(
    session_id: str,
    question_id: str,
    selected_answers: List[str],
    time_spent: Optional[int] = None
):
    """Submit answer for DVA-C02 question"""
    try:
        session = db.user_sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get practice set and question details
        practice_set = db.dva_practice_sets.find_one({"set_id": session["set_id"]})
        question = next(
            (q for q in practice_set["questions"] if q["question_id"] == question_id),
            None
        )
        
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        # Check correctness
        is_correct = set(selected_answers) == set(question["correct_answers"])
        
        # Store response
        response = {
            "user_id": session["user_id"],
            "session_id": session_id,
            "set_id": session["set_id"],
            "question_id": question_id,
            "selected_answers": selected_answers,
            "correct_answers": question["correct_answers"],
            "is_correct": is_correct,
            "is_scored": question["is_scored"],
            "domain": question["domain"],
            "difficulty": question["difficulty"],
            "time_spent": time_spent,
            "submitted_at": datetime.now()
        }
        
        db.dva_responses.insert_one(response)
        
        # Update session progress
        db.user_sessions.update_one(
            {"session_id": session_id},
            {"$inc": {"current_question_index": 1}}
        )
        
        return {
            "correct": is_correct,
            "is_scored": question["is_scored"],
            "question_number": session["current_question_index"] + 1,
            "next_question_available": session["current_question_index"] + 1 < 65,
            "message": "Answer submitted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skip-dva-question")
async def skip_dva_question(session_id: str, question_id: str):
    """Skip current DVA-C02 question"""
    try:
        session = db.user_sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Record skip
        skip_record = {
            "user_id": session["user_id"],
            "session_id": session_id,
            "set_id": session["set_id"],
            "question_id": question_id,
            "selected_answers": [],
            "correct_answers": [],
            "is_correct": False,
            "is_scored": True,  # Will be determined from question data
            "skipped": True,
            "submitted_at": datetime.now()
        }
        
        db.dva_responses.insert_one(skip_record)
        
        # Update session progress
        db.user_sessions.update_one(
            {"session_id": session_id},
            {"$inc": {"current_question_index": 1}}
        )
        
        return {
            "message": "Question skipped",
            "question_number": session["current_question_index"] + 1,
            "next_question_available": session["current_question_index"] + 1 < 65
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dva-exam-score")
async def get_dva_exam_score(session_id: str):
    """Get comprehensive DVA-C02 exam analytics and score"""
    try:
        session = db.user_sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all responses for this session
        responses = list(db.dva_responses.find({
            "session_id": session_id
        }))
        
        # Get practice set for domain analysis
        practice_set = db.dva_practice_sets.find_one({"set_id": session["set_id"]})
        
        # Calculate scores (only for scored questions)
        scored_responses = []
        for response in responses:
            # Find the question to check if it's scored
            question = next(
                (q for q in practice_set["questions"] if q["question_id"] == response["question_id"]),
                None
            )
            if question and question["is_scored"] and not response.get("skipped", False):
                scored_responses.append(response)
        
        total_scored = len([q for q in practice_set["questions"] if q["is_scored"]])
        correct_scored = len([r for r in scored_responses if r.get("is_correct", False)])
        
        # Calculate scaled score (AWS uses 100-1000 scale, 720 to pass)
        raw_percentage = (correct_scored / total_scored) * 100 if total_scored > 0 else 0
        scaled_score = int(100 + (raw_percentage / 100) * 900)
        passed = scaled_score >= 720
        
        # Domain-wise performance
        domain_stats = calculate_domain_performance(responses, practice_set)
        
        # Time analysis
        time_stats = calculate_time_statistics(responses, session)
        
        return {
            "exam_results": {
                "scaled_score": scaled_score,
                "raw_percentage": round(raw_percentage, 1),
                "passed": passed,
                "passing_score": 720,
                "result": "PASS" if passed else "FAIL",
                "grade": get_letter_grade(scaled_score)
            },
            "question_breakdown": {
                "total_questions": 65,
                "scored_questions": total_scored,
                "unscored_questions": 15,
                "answered": len([r for r in responses if not r.get("skipped", False)]),
                "correct": len([r for r in responses if r.get("is_correct", False)]),
                "skipped": len([r for r in responses if r.get("skipped", False)]),
                "incorrect": len([r for r in responses if not r.get("is_correct", False) and not r.get("skipped", False)])
            },
            "domain_performance": domain_stats,
            "time_analysis": time_stats,
            "exam_readiness": get_readiness_assessment(scaled_score, domain_stats),
            "recommendations": generate_study_recommendations(domain_stats),
            "session_info": {
                "session_id": session_id,
                "set_number": practice_set["set_number"],
                "started_at": session["started_at"],
                "completed_at": datetime.now()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dva-answer-explanation")
async def get_dva_answer_explanation_with_gemini(
    question_id: str,
    user_answers: List[str],
    explanation_type: str = "detailed"
):
    """Get detailed AI explanation using Gemini for DVA-C02 questions"""
    try:
        # Find question across all practice sets
        practice_set = db.dva_practice_sets.find_one({
            "questions.question_id": question_id
        })
        
        if not practice_set:
            raise HTTPException(status_code=404, detail="Question not found")
        
        question = next(
            (q for q in practice_set["questions"] if q["question_id"] == question_id),
            None
        )
        
        # Create detailed prompt for Gemini
        prompt = f"""
        As an AWS Certified Developer Associate expert, provide a comprehensive explanation for this exam question:
        
        QUESTION: {question['question']}
        OPTIONS: {', '.join(question['options'])}
        CORRECT ANSWER(S): {', '.join(question['correct_answers'])}
        USER'S ANSWER(S): {', '.join(user_answers)}
        DOMAIN: {question['domain'].title()}
        AWS SERVICES: {', '.join(question['aws_services'])}
        DIFFICULTY: {question['difficulty']}
        
        Please provide:
        1. **Why the correct answer is right**: Detailed technical explanation
        2. **Why other options are incorrect**: Specific reasons for each wrong option
        3. **Key AWS concepts**: Core developer concepts being tested
        4. **Code examples**: Relevant SDK code, CLI commands, or configuration
        5. **Best practices**: AWS development best practices related to this topic
        6. **Common mistakes**: What developers often get wrong in this area
        7. **Further reading**: Specific AWS documentation sections to study
        
        Focus on practical, hands-on knowledge that a certified AWS developer should possess.
        Include real-world scenarios and implementation details.
        """
        
        # Generate explanation using Gemini
        config = types.GenerateContentConfig(
            system_instruction="You are an expert AWS instructor helping developers understand certification exam concepts with practical, detailed explanations.",
            max_output_tokens=2000,
            temperature=0.3
        )
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=config
        )
        
        return {
            "question_id": question_id,
            "domain": question["domain"],
            "aws_services": question["aws_services"],
            "user_was_correct": set(user_answers) == set(question["correct_answers"]),
            "correct_answers": question["correct_answers"],
            "user_answers": user_answers,
            "detailed_explanation": response.text,
            "difficulty": question["difficulty"],
            "question_type": question["question_type"],
            "generated_by": "Gemini-2.0-Flash-Exp"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

# Helper Functions
def calculate_domain_performance(responses: List[dict], practice_set: dict) -> Dict:
    """Calculate performance by domain"""
    domain_stats = {}
    
    for domain in DVA_C02_DOMAINS.keys():
        domain_responses = []
        domain_questions = [q for q in practice_set["questions"] if q["domain"] == domain]
        
        for response in responses:
            if any(q["question_id"] == response["question_id"] and q["domain"] == domain 
                   for q in domain_questions):
                domain_responses.append(response)
        
        if domain_responses:
            correct = len([r for r in domain_responses if r.get("is_correct", False)])
            total = len(domain_responses)
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            domain_stats[domain] = {
                "total_questions": len(domain_questions),
                "answered": total,
                "correct": correct,
                "accuracy": round(accuracy, 1),
                "weight": DVA_C02_DOMAINS[domain]["weight"] * 100,
                "status": "Strong" if accuracy >= 80 else "Needs Improvement" if accuracy >= 60 else "Weak"
            }
    
    return domain_stats

def calculate_time_statistics(responses: List[dict], session: dict) -> Dict:
    """Calculate time-related statistics"""
    total_time_spent = sum(r.get("time_spent", 0) for r in responses if r.get("time_spent"))
    answered_questions = len([r for r in responses if not r.get("skipped", False)])
    avg_time_per_question = total_time_spent / answered_questions if answered_questions > 0 else 0
    
    return {
        "total_time_spent_minutes": round(total_time_spent / 60, 1),
        "average_time_per_question_seconds": round(avg_time_per_question, 1),
        "time_remaining": calculate_time_remaining(session),
        "pace": "Good" if avg_time_per_question <= 120 else "Slow" if avg_time_per_question <= 180 else "Too Slow"
    }

def get_letter_grade(scaled_score: int) -> str:
    """Convert scaled score to letter grade"""
    if scaled_score >= 900:
        return "A+"
    elif scaled_score >= 850:
        return "A"
    elif scaled_score >= 800:
        return "A-"
    elif scaled_score >= 750:
        return "B+"
    elif scaled_score >= 720:
        return "B"
    elif scaled_score >= 650:
        return "C"
    else:
        return "F"

def get_readiness_assessment(scaled_score: int, domain_stats: Dict) -> str:
    """Assess exam readiness based on performance"""
    if scaled_score >= 800:
        return "Excellent - Well prepared for the exam"
    elif scaled_score >= 720:
        return "Good - Ready for the exam with minor review"
    elif scaled_score >= 650:
        return "Fair - Need more preparation in weak domains"
    else:
        return "Poor - Significant study required before attempting exam"

def generate_study_recommendations(domain_stats: Dict) -> List[str]:
    """Generate study recommendations based on domain performance"""
    recommendations = []
    
    for domain, stats in domain_stats.items():
        if stats["accuracy"] < 70:
            domain_config = DVA_C02_DOMAINS[domain]
            recommendations.append(
                f"Focus on {domain.title()} domain - review {', '.join(domain_config['services'])} services"
            )
    
    return recommendations

async def end_session_due_to_timeout(session_id: str):
    """End session when time expires"""
    db.user_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"is_completed": True, "ended_reason": "timeout"}}
    )


















@app.get("/health")
async def health_check():
    """Health check including Gemini connectivity"""
    try:
        # Test Gemini connectivity
        test_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Hello, respond with 'OK' if you're working."
        )
        gemini_status = "connected" if "OK" in test_response.text else "error"
    except:
        gemini_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "AWS DVA-C02 Practice Exam API with Gemini",
        "version": "2.0",
        "gemini_status": gemini_status,
        "gemini_model": "gemini-2.0-flash-exp",
        "features": ["25 unique practice sets", "65 questions per set", "Official exam format", "AI explanations"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
