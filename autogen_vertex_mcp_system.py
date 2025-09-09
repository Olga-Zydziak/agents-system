# autogen_vertex_mcp_system.py
"""
AutoGen Multi-Agent System with Vertex AI Search as MCP Tool
Agents learn from mission memory without prompt injection
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import autogen
from autogen import AssistantAgent, UserProxyAgent
from google.cloud import discoveryengine_v1 as de
from google.cloud import storage
from google.api_core.client_options import ClientOptions

# ======================== Configuration ========================


@dataclass
class SystemConfig:
    """Central configuration"""

    vertex_search_config: str = (
        "projects/815755318672/locations/us/collections/default_collection/dataStores/external-memory-connector_1756845276280_gcs_store/servingConfigs/default_config"
    )
    vertex_location: str = "us"

    # AutoGen config - możesz zmienić na Gemini jeśli masz setup
    autogen_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "config_list": [
                {
                    "model": "gpt-4-turbo-preview",  # lub "gemini-1.5-pro" jeśli używasz Vertex AI
                    "api_key": "your-api-key",  # lub vertex ai credentials
                    "temperature": 0.7,
                }
            ],
            "timeout": 120,
            "cache_seed": 42,
        }
    )

    max_search_results: int = 5
    enable_learning: bool = True


# ======================== Vertex AI Search Tool ========================


TOOLS_SPEC = [
    {
        "name": "search_mission_memory",
        "description": "Search past missions and return a compact list of matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": ["string", "object"]},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_mission_plan",
        "description": "Fetch full plan JSON for a mission id.",
        "parameters": {
            "type": "object",
            "properties": {"mission_id": {"type": "string"}},
            "required": ["mission_id"],
        },
    },
    {
        "name": "analyze_patterns",
        "description": "Aggregate patterns over cached search results.",
        "parameters": {
            "type": "object",
            "properties": {"min_score": {"type": "number", "default": 0.0}},
        },
    },
]


class VertexSearchTool:
    """Vertex AI Search as a tool for AutoGen agents"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.storage_client = storage.Client()
        self.search_client = de.SearchServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{config.vertex_location}-discoveryengine.googleapis.com"
            )
        )
        self._cache = {}

    def search_mission_memory(
        self, query: str, filters: Optional[str] = None, top_k: int = 5
    ) -> str:
        """
        Search mission memories. Returns JSON string.
        This is the function that AutoGen agents will call.
        """
        try:
            # Check cache
            cache_key = f"{query}_{filters}_{top_k}"
            if cache_key in self._cache:
                return json.dumps(self._cache[cache_key])

            # Build request
            req = de.SearchRequest(
                serving_config=self.config.vertex_search_config,
                query=query,
                page_size=min(top_k, self.config.max_search_results),
                filter=filters if filters else None,
            )

            # Execute search
            results = []
            for r in self.search_client.search(request=req):
                doc = r.document
                sdata = doc.struct_data or {}

                results.append(
                    {
                        "mission_id": sdata.get("mission_id", ""),
                        "score": sdata.get("final_score", 0),
                        "approved": sdata.get("approved", False),
                        "nodes_count": sdata.get("nodes_count", 0),
                        "edges_count": sdata.get("edges_count", 0),
                        "has_optimization": sdata.get("has_optimization", False),
                        "has_rollback": sdata.get("has_rollback", False),
                        "has_retry": sdata.get("has_retry", False),
                        "tags": sdata.get("tags", []),
                        "links": sdata.get("links", {}),
                    }
                )

            response = {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
            }

            # Cache result
            self._cache[cache_key] = response

            return json.dumps(response, ensure_ascii=False)

        except Exception as e:
            error_response = {
                "status": "error",
                "error": str(e),
                "query": query,
                "results": [],
            }
            return json.dumps(error_response)

    def get_mission_plan(self, mission_id: str) -> str:
        """
        Get complete plan for a mission. Returns JSON string.
        """
        try:
            # First find the mission
            search_result = self.search_mission_memory(
                query=f"mission_id:{mission_id}", top_k=1
            )
            search_data = json.loads(search_result)

            if not search_data.get("results"):
                return json.dumps({"status": "not_found", "mission_id": mission_id})

            links = search_data["results"][0].get("links", {})
            plan_uri = links.get("plan_uri")

            if not plan_uri or not plan_uri.startswith("gs://"):
                return json.dumps({"status": "no_plan", "mission_id": mission_id})

            # Fetch from GCS
            bucket_name, _, path = plan_uri[5:].partition("/")
            blob = self.storage_client.bucket(bucket_name).blob(path)
            plan_content = blob.download_as_text(encoding="utf-8")
            plan_data = json.loads(plan_content)

            return json.dumps(
                {"status": "success", "mission_id": mission_id, "plan": plan_data}
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "error": str(e), "mission_id": mission_id}
            )

    def analyze_patterns(self, min_score: float = 90.0) -> str:
        """
        Analyze successful patterns. Returns JSON string.
        """
        try:
            # Search for high-scoring approved missions
            results_json = self.search_mission_memory(
                query="approved:true", filters=f"final_score >= {min_score}", top_k=20
            )
            results_data = json.loads(results_json)

            if not results_data.get("results"):
                return json.dumps({"status": "no_data", "patterns": {}})

            results = results_data["results"]
            total = len(results)

            patterns = {
                "optimization_rate": sum(
                    1 for r in results if r.get("has_optimization")
                )
                / total,
                "rollback_rate": sum(1 for r in results if r.get("has_rollback"))
                / total,
                "retry_rate": sum(1 for r in results if r.get("has_retry")) / total,
                "avg_nodes": sum(r.get("nodes_count", 0) for r in results) / total,
                "avg_edges": sum(r.get("edges_count", 0) for r in results) / total,
                "avg_score": sum(r.get("score", 0) for r in results) / total,
                "total_analyzed": total,
            }

            # Find common tags
            tag_counts = {}
            for r in results:
                for tag in r.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            patterns["common_tags"] = sorted(
                tag_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return json.dumps({"status": "success", "patterns": patterns})

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e), "patterns": {}})


# ======================== AutoGen Agents Setup ========================


class MissionPlanningTeam:
    """AutoGen multi-agent team for mission planning"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.vertex_tool = VertexSearchTool(config)
        self.setup_agents()

    def setup_agents(self):
        """Setup AutoGen agents with tools"""

        # Memory Analyst Agent - analizuje pamięć misji
        self.memory_analyst = AssistantAgent(
            name="MemoryAnalyst",
            system_message="""You are a memory analyst agent. Your role is to:
            1. Search the mission memory for relevant past experiences
            2. Identify successful patterns from previous missions
            3. Extract key learnings and insights
            
            
            You MUST call the provided tools instead of describing intentions.
            -Step 1: call search_mission_memory; Step 2: call get_mission_plan for each hit;
            -Step 3: optionally call analyze_patterns. Never fabricate memory results.
            -If a tool fails, simplify and try again.
            
            You have access to these functions:
            - search_mission_memory(query, filters, top_k): Search past missions
            - get_mission_plan(mission_id): Get detailed plan for a mission
            - analyze_patterns(min_score): Analyze successful patterns
            
            Always base recommendations on data from the memory search.
            Return findings as structured JSON when possible.""",
            llm_config=self.config.autogen_config,
            function_map={
                "search_mission_memory": self.vertex_tool.search_mission_memory,
                "get_mission_plan": self.vertex_tool.get_mission_plan,
                "analyze_patterns": self.vertex_tool.analyze_patterns,
            },
        )

        # Graph Designer Agent - projektuje grafy wykonania
        self.graph_designer = AssistantAgent(
            name="GraphDesigner",
            system_message="""You are a graph execution plan designer. Your role is to:
            1. Design execution graphs with nodes and edges
            2. Incorporate patterns learned from successful missions
            3. Ensure robustness with error handling, rollback, and optimization
            
            Create plans in this JSON format:
            {
                "entry_point": "StartNode",
                "nodes": [
                    {"name": "NodeName", "implementation": "function", "params": {}}
                ],
                "edges": [
                    {"from": "Node1", "to": "Node2", "condition": "on_success"}
                ]
            }
            
            Use insights from MemoryAnalyst to improve your designs.""",
            llm_config=self.config.autogen_config,
        )

        # Quality Critic Agent - ocenia i ulepsza plany
        self.quality_critic = AssistantAgent(
            name="QualityCritic",
            system_message="""You are a quality critic. Your role is to:
            1. Evaluate proposed plans against historical success patterns
            2. Identify missing robustness features (rollback, retry, optimization)
            3. Suggest improvements based on data from mission memory
            4. Calculate confidence scores
            Use memory tools to justify critique.
            Call search_mission_memory and analyze_patterns; cite concrete results.
            Be constructive but critical. Use data to support your assessments.""",
            llm_config=self.config.autogen_config,
            function_map={
                "search_mission_memory": self.vertex_tool.search_mission_memory,
                "analyze_patterns": self.vertex_tool.analyze_patterns,
            },
        )

        # Coordinator - nie wykonuje kodu, tylko zarządza przepływem
        self.coordinator = UserProxyAgent(
            name="Coordinator",
            system_message="Coordinate the planning process.",
            code_execution_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

    async def create_mission_plan(self, mission_prompt: str) -> Dict[str, Any]:
        """
        Create a mission plan using the multi-agent team
        """
        logging.info(f"Starting mission planning for: {mission_prompt}")

        # Create group chat
        groupchat = autogen.GroupChat(
            agents=[
                self.coordinator,
                self.memory_analyst,
                self.graph_designer,
                self.quality_critic,
            ],
            messages=[],
            max_round=14,
            speaker_selection_method="round_robin",
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=self.config.autogen_config
        )

        # Start the planning process
        initial_message = f"""
        Create an execution plan for this mission: {mission_prompt}
        Rules:
        - Use tool calls. DO NOT describe intended calls in text.
        - Step 1: CALL search_mission_memory with concise English query, top_k=3–5.
        - Step 2: For top hits, CALL get_mission_plan(mission_id) and extract nodes/edges.
        - Step 3: CALL analyze_patterns(min_score ~0.7) if helpful.
        - If a tool errors, retry with simpler args.
        
        Process:
        1. MemoryAnalyst: Search for similar successful missions
        2. MemoryAnalyst: Get FULL DETAILS (complete plans) of top 2-3 similar missions
        3. MemoryAnalyst: Analyze the exact structure - what nodes, edges, conditions they use
        4. GraphDesigner: Create a plan based on ACTUAL successful plan structures
        5. QualityCritic: Compare new plan with successful ones and suggest improvements
        
        MemoryAnalyst, start by finding similar missions and then GET THEIR COMPLETE PLANS.
        """

        # Initiate chat
        await self.coordinator.a_initiate_chat(
            manager, message=initial_message, clear_history=True
        )

        # Extract the final plan from conversation
        final_plan = self._extract_plan_from_messages(groupchat.messages)

        # Get learning context
        learning_context = self._extract_learning_context(groupchat.messages)

        return {
            "mission_prompt": mission_prompt,
            "final_plan": final_plan,
            "learning_context": learning_context,
            "conversation_rounds": len(groupchat.messages),
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_plan_from_messages(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract the final plan from agent messages"""
        # Look for JSON plans in reverse order (latest first)
        for msg in reversed(messages):
            if msg.get("name") == "GraphDesigner":
                content = msg.get("content", "")
                try:
                    # Try to find JSON in the content
                    import re

                    json_match = re.search(
                        r'\{.*"entry_point".*"nodes".*"edges".*\}', content, re.DOTALL
                    )
                    if json_match:
                        return json.loads(json_match.group())
                except:
                    continue

        # Return empty plan if not found
        return {"entry_point": "", "nodes": [], "edges": []}

    def _extract_learning_context(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract what was learned from memory"""
        context = {
            "searched_queries": [],
            "analyzed_missions": [],
            "identified_patterns": {},
            "applied_improvements": [],
        }

        for msg in messages:
            if msg.get("name") == "MemoryAnalyst":
                content = msg.get("content", "")
                # Extract search queries and results
                if "search_mission_memory" in content:
                    context["searched_queries"].append(content[:100])
                if "mission_id" in content:
                    # Extract mission IDs mentioned
                    import re

                    ids = re.findall(r"mission_\w+", content)
                    context["analyzed_missions"].extend(ids[:5])

            elif msg.get("name") == "QualityCritic":
                if "improvement" in msg.get("content", "").lower():
                    context["applied_improvements"].append(msg.get("content", "")[:200])

        return context


# ======================== Main Execution ========================


class MissionExecutor:
    """Main system orchestrator"""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.team = MissionPlanningTeam(self.config)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    async def execute_mission(self, mission_prompt: str) -> Dict[str, Any]:
        """Execute a mission with learning from memory"""

        logging.info("=" * 60)
        logging.info(f"MISSION: {mission_prompt}")
        logging.info("=" * 60)

        # Create plan using multi-agent team
        result = await self.team.create_mission_plan(mission_prompt)

        # Log what was learned
        if result.get("learning_context", {}).get("analyzed_missions"):
            logging.info(
                f"Learned from missions: {result['learning_context']['analyzed_missions']}"
            )

        logging.info(
            f"Plan created with {len(result['final_plan'].get('nodes', []))} nodes"
        )

        return result

    def run(self, mission_prompt: str) -> Dict[str, Any]:
        """Synchronous wrapper for execute_mission"""
        return asyncio.run(self.execute_mission(mission_prompt))
