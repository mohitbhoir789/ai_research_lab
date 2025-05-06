"""
Guardrails Module
Implements safety checks and domain restrictions for the AI Research Assistant.
Uses Gemini for more sophisticated topic detection.
"""

import re
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class GuardrailsChecker:
    """Enforces safety checks and domain restrictions using LLM for topic detection."""

    def __init__(self):
        # Keep the allowed domains for reference and documentation
        self.allowed_domains = {
            "computer_science": [
                "algorithms", "data structures", "programming", "software engineering",
                "databases", "operating systems", "networks", "security", "cloud computing",
                "distributed systems", "quantum computing", "artificial intelligence",
                "machine learning", "deep learning", "computer vision", "nlp",
                "robotics", "web development", "systems", "architecture",
                "gradient descent", "optimization", "neural network", "backpropagation",
                "reinforcement learning", "transformer", "attention", "stochastic",
                "generative model", "sentiment analysis", "natural language processing",
                "text mining", "information retrieval", "recommendation systems"
            ],
            "data_science": [
                "statistics", "data analysis", "data mining", "data visualization",
                "big data", "data engineering", "data modeling", "machine learning",
                "deep learning", "neural networks", "predictive analytics",
                "business intelligence", "data warehousing", "etl", "data pipelines",
                "experimentation", "ab testing", "clustering", "regression", "classification",
                "gradient descent", "optimization", "loss function", "hyperparameter",
                "algorithm", "feature engineering", "dimensionality", "model training",
                "backpropagation", "weights", "sgd", "adam", "rmsprop", "momentum",
                "sentiment analysis", "text analytics", "natural language processing",
                "opinion mining", "emotion detection", "data extraction", "information extraction"
            ]
        }
        
        # Initialize Gemini client
        self.gemini_client = None
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai
            logger.info("Gemini client initialized for topic detection")
        else:
            logger.warning("GEMINI_API_KEY not found. Falling back to simple keyword matching for topic detection.")
            
        # Keep OpenAI client as fallback
        self.openai_client = None
        self.async_openai_client = None
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.async_openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized as fallback")

    async def check_input(self, text: str) -> Dict[str, Any]:
        """
        Validate user input against safety rules and domain restrictions.
        Uses LLM to detect if the topic is relevant to CS or DS.
        
        Args:
            text: User input text
            
        Returns:
            Dict with validation results
        """
        # Check for appropriate length
        if len(text) > 2000:
            return {
                "passed": False,
                "reason": "too_long",
                "message": "Please keep your queries under 2000 characters."
            }
        
        # Check for code safety (no execution commands)
        text_lower = text.lower()
        dangerous_patterns = [
            r"system\s*\(", r"exec\s*\(", r"eval\s*\(",
            r"os\.", r"subprocess\.", r"bash", r"shell"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower):
                return {
                    "passed": False,
                    "reason": "unsafe_code",
                    "message": "I cannot process potentially unsafe code execution commands."
                }
                
        # Check for domain relevance using LLM
        is_relevant, method_used = await self._check_topic_relevance(text)
        
        if not is_relevant:
            logger.info(f"Topic rejected as off-topic. Method used: {method_used}. Text: '{text[:100]}...'")
            return {
                "passed": False,
                "reason": "off_topic",
                "message": "I can only assist with Computer Science and Data Science topics. Please rephrase your question to focus on these domains."
            }

        logger.info(f"Topic accepted as relevant. Method used: {method_used}")
        return {"passed": True}
        
    async def _check_topic_relevance(self, text: str) -> tuple[bool, str]:
        """
        Check if the text is relevant to Computer Science or Data Science domains.
        Uses Gemini if available, falls back to keyword matching or OpenAI.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_relevant, method_used)
        """
        # Always consider common educational/research phrases as relevant
        educational_phrases = ["what is", "how does", "explain", "define", "tell me about", 
                               "describe", "research on", "study of", "concept of", "application of",
                               "example of", "tutorial", "guide", "learn", "understand"]
        
        text_lower = text.lower()
        
        # Check if this is a basic educational question
        for phrase in educational_phrases:
            if phrase in text_lower:
                # For educational questions, do a more lenient check
                # Attempting to detect if this might be related to CS/DS
                potential_cs_ds_terms = [
                    "algorithm", "data", "program", "code", "software", "hardware", 
                    "computer", "network", "system", "machine", "learning", "ai", 
                    "artificial intelligence", "analysis", "model", "predict", 
                    "statistic", "neural", "deep", "mining", "processing",
                    "sentiment", "language", "classification", "recognition",
                    "clustering", "regression", "inference", "computation", "database",
                    "cloud", "security", "encryption", "web", "internet", "api",
                    "framework", "visualization", "dashboard"
                ]
                
                for term in potential_cs_ds_terms:
                    if term in text_lower:
                        # If it looks like an educational question about a CS/DS term,
                        # allow it through without requiring LLM check
                        logger.info(f"Educational question with CS/DS term detected: '{term}'")
                        return True, "educational_pattern"
        
        # Try with Gemini first
        if self.gemini_client:
            try:
                # Use Gemini to classify the topic with improved prompt
                prompt = """You are an AI that determines if a query is related to Computer Science or Data Science. 
Computer Science includes:
Programming, Algorithms, Databases, Networks, Security, Artificial Intelligence, Machine Learning, Computer Vision, NLP, Web Development, Cloud Computing, Operating Systems, Software Engineering, Data Structures, Distributed Systems, Quantum Computing, Robotics, Sentiment Analysis, Information Retrieval, Recommendation Systems, Theory of Computation, Automata Theory, Formal Languages, Computability Theory, Complexity Theory, Computer Architecture, Digital Logic, Processor Design, Memory Hierarchies, Parallel Architectures, Embedded Systems, Programming Languages, Compiler Design, Programming Paradigms, Language Semantics, Human-Computer Interaction (HCI), User Interface Design, Usability, Interaction Design, Graphics and Visualization, Computer Graphics, Scientific Visualization, Mobile Computing, Mobile Application Development, Wireless Networks, Formal Methods, Verification, Model Checking, Theorem Proving, Cryptography, Encryption, Decryption, Secure Protocols, Parallel and Distributed Computing, Concurrency, Distributed Algorithms, Bioinformatics (Computational aspects), Computational Science, Computational Physics, Computational Chemistry, Software Testing and Quality Assurance, Requirements Engineering, System Administration, Network Security, Compilers, Operating Systems Design, Programming Language Theory, Computer Graphics, Simulation, Optimization, Formal Verification, Computer Architecture Design, Network Protocols, Cybersecurity, Ethical Hacking, Digital Forensics, Game Development.

Data Science includes:
Statistics, Data Analysis, Data Mining, Data Visualization, Big Data, Data Engineering, Data Modeling, Machine Learning, Prediction, Clustering, Classification, Regression, Neural Networks, Feature Engineering, A/B Testing, Sentiment Analysis, Text Analytics, NLP, Statistical Modeling, Regression Analysis, Time Series Analysis, Bayesian Statistics, Hypothesis Testing, Experimental Design, Econometrics, Operations Research, Optimization, Simulation, Actuarial Science (Data-related aspects), Geospatial Data Analysis, Survival Analysis, Causal Inference, Data Governance, Data Ethics, Big Data Technologies, Distributed File Systems (HDFS), NoSQL Databases, Distributed Processing Frameworks (Spark, Hadoop), Data Warehousing, ETL (Extract, Transform, Load), Business Intelligence, Experimentation, Feature Selection, Model Evaluation and Selection, Bias and Fairness in AI, Predictive Analytics, Descriptive Analytics, Prescriptive Analytics, Data Cleaning and Preprocessing, Exploratory Data Analysis (EDA), Statistical Inference, ANOVA, Dimensionality Reduction, Principal Component Analysis (PCA), Independent Component Analysis (ICA), Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Gradient Boosting, Ensemble Methods, Deep Learning, Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers (in NLP), Data Pipelines, Data Architecture, Cloud Data Platforms, AWS S3, Google Cloud Storage, Azure Data Lake Storage, Big Data Processing Frameworks (Apache Spark, Apache Flink), Data Warehouses (Snowflake, Amazon Redshift, Google BigQuery), NoSQL Databases (MongoDB, Cassandra), Data Lakes, DataOps, MLOps (Machine Learning Operations), Business Analytics, Customer Analytics, Financial Analytics, Healthcare Analytics, Marketing Analytics, Web Analytics, Data Storytelling, Dashboarding, BI Tools (Tableau, Power BI), Statistical Software (R, SAS, SPSS), Programming Languages (Python, R, SQL, Scala, Julia), Libraries and Frameworks (Pandas, NumPy, SciPy, Scikit-learn, TensorFlow, PyTorch, Keras, Spark MLlib               

The user query is: "{query}"

Questions about definitions, applications, or techniques within these fields are relevant even if they seem basic.
First, think step by step about whether this query is related to Computer Science or Data Science.
Then respond in JSON format with two fields:
1. "is_relevant": true or false
2. "reasoning": brief explanation of why the topic is or isn't relevant
"""

                # Configure generation parameters
                generation_config = {
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 150,
                }
                
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                # Create the model and generate response
                gemini_model = self.gemini_client.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Format the prompt with the actual query text
                formatted_prompt = prompt.format(query=text)
                
                # Generate the response
                response = gemini_model.generate_content(formatted_prompt)
                
                # Parse the response - extract JSON from it
                content = response.text.strip()
                logger.debug(f"Gemini response: {content}")
                
                # Try to extract JSON from the response
                try:
                    # Check if the response contains JSON-like structure
                    if "{" in content and "}" in content:
                        # Extract the JSON part
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        answer_json = json.loads(json_str)
                    else:
                        # Try to parse the whole response as JSON
                        answer_json = json.loads(content)
                        
                    is_relevant = answer_json.get("is_relevant", False)
                    reasoning = answer_json.get("reasoning", "No reasoning provided")
                    
                    logger.info(f"Gemini topic classification: {is_relevant}, reason: {reasoning[:50]}...")
                    return is_relevant, "gemini_classification"
                except json.JSONDecodeError:
                    # If not valid JSON, check for keyword indicators
                    is_relevant = ("relevant" in content.lower() and not "not relevant" in content.lower()) or "is relevant" in content.lower()
                    reasoning = content
                    
                    logger.info(f"Gemini text classification (non-JSON): {is_relevant}")
                    return is_relevant, "gemini_text_classification"
                
            except Exception as e:
                logger.error(f"Error using Gemini for topic detection: {str(e)}")
                # Fall back to OpenAI or keyword matching
        
        # Fall back to OpenAI if available
        if self.async_openai_client:
            try:
                # Same OpenAI logic as before, but as a fallback
                logger.info("Falling back to OpenAI for topic classification")
                messages = [
                    {"role": "system", "content": "You determine if a query relates to Computer Science or Data Science."},
                    {"role": "user", "content": f"Is this query related to Computer Science or Data Science? Query: '{text}'. Answer yes or no and explain briefly."}
                ]
                
                response = await self.async_openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=100
                )
                
                content = response.choices[0].message.content.strip().lower()
                is_relevant = "yes" in content[:10] or "related" in content[:20]
                
                logger.info(f"OpenAI fallback classification: {is_relevant}")
                return is_relevant, "openai_fallback_classification"
                
            except Exception as e:
                logger.error(f"Error using OpenAI fallback for topic detection: {str(e)}")
        
        # Ultimate fallback to keyword matching
        is_relevant = self._expanded_keyword_match(text)
        return is_relevant, "expanded_keyword_fallback"

    def _expanded_keyword_match(self, text: str) -> bool:
        """
        Enhanced keyword matching with better coverage of CS/DS topics.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text contains relevant keywords, False otherwise
        """
        text_lower = text.lower()
        
        # Start with existing domain keywords
        all_keywords = set()
        for keywords in self.allowed_domains.values():
            all_keywords.update(keywords)
        
        # Add more general CS/DS terms that might not be in the specific lists
        additional_keywords = [
            "sentiment analysis", "nlp", "natural language", "text analysis",
            "algorithm", "compute", "computing", "computer", "program", "code",
            "software", "api", "application", "web", "network", "server",
            "database", "query", "system", "interface", "framework", "library",
            "function", "variable", "class", "object", "method", "git", "cloud",
            "data", "analytics", "metrics", "measure", "insight", "trend",
            "prediction", "forecast", "model", "train", "test", "validate",
            "learn", "neural", "network", "deep", "gradient", "computation",
            "processing", "information", "technology", "IT", "tech", "language"
        ]
        
        all_keywords.update(additional_keywords)
        
        # Check for keyword matches
        for keyword in all_keywords:
            # Check for exact matches
            if keyword in text_lower:
                logger.info(f"Keyword matched: {keyword}")
                return True
            
            # For multi-word keywords, check if all words are present
            if " " in keyword:
                words = keyword.split()
                if all(word in text_lower for word in words):
                    logger.info(f"Multi-word keyword matched: {keyword}")
                    return True
        
        return False

    def sanitize_output(self, text: str) -> str:
        """
        Sanitize agent outputs for safety and relevance.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove any potentially unsafe commands or URLs
        text = re.sub(r"(system|exec|eval)\s*\([^)]*\)", "[REMOVED]", text)
        
        # Ensure proper markdown formatting
        text = self._fix_markdown(text)
        
        return text

    def _fix_markdown(self, text: str) -> str:
        """
        Fix common markdown formatting issues.
        
        Args:
            text: Text to fix
            
        Returns:
            Fixed text
        """
        # Ensure code blocks are properly formatted
        text = re.sub(r'```(\w+)?\s*\n', r'```\1\n', text)
        text = re.sub(r'\n\s*```', r'\n```', text)
        
        # Ensure headers have space after #
        text = re.sub(r'(#+)([^\s])', r'\1 \2', text)
        
        return text