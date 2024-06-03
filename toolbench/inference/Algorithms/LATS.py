import os
import pickle
import time
from operator import itemgetter
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
import math
from typing import Literal
import json
from collections import defaultdict, deque
from langchain.tools import StructuredTool
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

import requests
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain as as_runnable, RunnablePassthrough
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from langchain.tools.render import render_text_description

from langgraph.graph import END, StateGraph
from toolbench.inference.Algorithms.base_search import base_search_method


class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, super fluency,"
                    " and general quality of the response. Do not be longer than 256 characters."
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
            self,
            messages: list[BaseMessage],
            reflection: Reflection,
            parent=None
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection

        if parent is None:
            self.depth = 1
        else:
            self.depth = parent.depth + 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)
        for m in self.get_messages():
            print(m.content)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        """Select the child with the highest UCT to search next."""
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

    def to_json(self):
        json_obj = {}
        json_obj["depth"] = self.depth
        json_obj["tool_calls"] = []
        json_obj["reflection"] = []
        json_obj["thought"] = []
        json_obj["child_count"] = len(self.children)
        for message in self.get_messages():

            if isinstance(message, ToolMessage):
                json_obj["tool_calls"].append(message.content)
            if isinstance(message, HumanMessage):
                json_obj["reflection"].append(message.content)
            if isinstance(message, AIMessage):
                json_obj["thought"].append(message.content)

        return json_obj

    def to_json_recursive(self):
        js_obj = self.to_json()
        js_obj["children"] = []
        for child in self.children:
            js_obj["children"].append(child.to_json_recursive())
        return js_obj


class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str


class LATSSearch(base_search_method):
    def __init__(self, llm, io_func, process_id=0, callbacks=None):
        super(LATSSearch, self).__init__(llm, io_func, process_id, callbacks)
        self.io_func = io_func
        self.process_id = process_id
        self.callbacks = callbacks if callbacks is not None else []

        os.environ["OPENAI_API_KEY"] = "sk-C8a9Ux05NoROx7jx4a2f858f325c4f1f84E5C12915A8E73b"
        os.environ["OPENAI_API_BASE"] = "https://api.chsdw.top/v1"

        self.llm = ChatOllama(model="llama3", format="json")
        self.reflection_llm = ChatOpenAI(model="gpt-4o")

        self.tools = self.make_tools()
        self.rendered_tools = render_text_description(self.tools)
        self.tool_executor = ToolExecutor(tools=self.tools)
        self.root = None

        tool_descriptions = self.rendered_tools.replace("{", "(").replace("}", ")")

        system_prompt = """You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        def printer(i):
            print(i)
            return i

        self.initial_answer_chain = self.prompt_template | printer | self.llm.with_config(
            run_name="GenerateInitialCandidate"
        )

        self.expansion_chain = self.prompt_template | self.generate_candidates

        self.parser = JsonOutputParser()

        self.graph = self.create_graph()

        @as_runnable
        def reflection_chain(inputs) -> Reflection:
            reflection_chain = self.get_ref_chain()
            tool_choices = reflection_chain.invoke(inputs)
            reflection = tool_choices[0]
            if not isinstance(inputs["candidate"][-1], AIMessage):
                reflection.found_solution = False
            return reflection

        self.reflection_chain = reflection_chain

    def get_ref_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Reflect and grade the assistant response to the user question below.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="candidate"),
            ]
        )

        reflection_llm_chain = (
                prompt
                | self.llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
            run_name="Reflection"
        )
                | PydanticToolsParser(tools=[Reflection])
        )
        return reflection_llm_chain

    def make_tools(self):
        tools = []
        for k, function in enumerate(self.io_func.functions[:-1]):
            toolbench_key = "D09PdWsx4fqLO2kzr6bEQ3YHXcAmg7ijtUu1p8TlR5oJnBGZVa"
            pure_api_name = self.io_func.api_name_reflect[function["name"]]

            def generate_tool_function(k, pure_api_name):
                def tool_function(action_input):
                    payload = {
                        "category": self.io_func.cate_names[k],
                        "tool_name": self.io_func.tool_names[k],
                        "api_name": pure_api_name,
                        "tool_input": action_input,
                        "strip": "truncate",
                        "toolbench_key": toolbench_key
                    }
                    print(f"api_name:{pure_api_name} tool_input:{action_input}")
                    headers = {"toolbench_key": toolbench_key}
                    response = requests.post("http://8.218.239.54:8080/rapidapi", json=payload, headers=headers,
                                             timeout=15)
                    if response.status_code != 200:
                        return json.dumps(
                            {"error": f"request invalid, data error. status_code={response.status_code}",
                             "response": ""}), 12
                    try:
                        response = response.json()
                    except Exception as e:
                        return json.dumps({"error": f"request invalid, data error: {str(e)}", "response": ""}), 12
                    if response["error"] == "API not working error...":
                        status_code = 6
                    elif response["error"] == "Unauthorized error...":
                        status_code = 7
                    elif response["error"] == "Unsubscribed error...":
                        status_code = 8
                    elif response["error"] == "Too many requests error...":
                        status_code = 9
                    elif response["error"] == "Rate limit per minute error...":
                        print("Reach api calling limit per minute, sleeping...")
                        time.sleep(10)
                        status_code = 10
                    elif response["error"] == "Message error...":
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code

                return tool_function

            tool_function = generate_tool_function(k, pure_api_name)
            prompt = "The parameters of this function are:"
            tool = StructuredTool.from_function(name=function["name"],
                                                description=(function["description"] + prompt + str(
                                                    function["parameters"]))[:1024],
                                                func=tool_function)
            tools.append(tool)

        return tools

    def generate_initial_response(self, state: TreeState) -> dict:
        """Generate the initial candidate response."""

        res = self.initial_answer_chain.invoke({"input": state["input"]})
        parsed = self.parser.invoke(res)

        # tool_responses = self.tool_executor.batch(
        #     [ToolInvocation(tool=r["type"],
        #                     tool_input=r["args"] if 'action_input' in r["args"] else {'action_input': {}}) for r in
        #      parsed]
        # )
        # output_messages = [res] + [
        #     ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
        #     for resp, tool_call in zip(tool_responses, parsed)
        # ]

        def call_tools(msg):
            """Simple sequential tool calling helper."""

            tool_map = {tool.name: tool for tool in self.tools}
            tool_calls = msg.tool_calls.copy()
            for tool_call in tool_calls:
                tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
            return tool_calls

        output_messages = [res] + [
            ToolMessage(content=output)
        ]

        reflection = self.reflection_chain.invoke(
            {"input": state["input"], "candidate": output_messages}
        )
        root = Node(output_messages, reflection=reflection)
        return {
            **state,
            "root": root,
        }

    def generate_candidates(self, messages: ChatPromptValue, config: RunnableConfig):
        n = config["configurable"].get("N", 3)
        bound_kwargs = self.llm.bind_tools(tools=self.tools, tool_choice="auto").kwargs
        chat_result = self.llm.generate(
            [messages.to_messages()],
            n=n,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
            **bound_kwargs,
        )
        return [gen.message for gen in chat_result.generations[0]]

    def expand(self, state: TreeState, config: RunnableConfig) -> dict:
        """Starting from the "best" node in the tree, generate N candidates for the next step."""
        root = state["root"]
        best_candidate: Node = root.best_child if root.children else root
        messages = best_candidate.get_trajectory()
        # Generate N candidates from the single child candidate
        new_candidates = self.expansion_chain.invoke(
            {"input": state["input"], "messages": messages}, config
        )
        parsed = self.parser.batch(new_candidates)
        flattened = [
            (i, tool_call)
            for i, tool_calls in enumerate(parsed)
            for tool_call in tool_calls
        ]
        tool_responses = self.tool_executor.batch(
            [
                ToolInvocation(tool=tool_call["type"],
                               tool_input=tool_call["args"] if 'action_input' in tool_call["args"] else {
                                   'action_input': {}})
                for _, tool_call in flattened
            ]
        )
        collected_responses = defaultdict(list)
        for (i, tool_call), resp in zip(flattened, tool_responses):
            collected_responses[i].append(
                ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
            )
        output_messages = []
        for i, candidate in enumerate(new_candidates):
            output_messages.append([candidate] + collected_responses[i])

        # Reflect on each candidate
        # For tasks with external validation, you'd add that here.
        reflections = self.reflection_chain.batch(
            [{"input": state["input"], "candidate": msges} for msges in output_messages],
            config,
        )
        # Grow tree
        child_nodes = [
            Node(cand, parent=best_candidate, reflection=reflection)
            for cand, reflection in zip(output_messages, reflections)
        ]
        best_candidate.children.extend(child_nodes)
        # We have already extended the tree directly, so we just return the state
        return state

    def create_graph(self):
        def should_loop(state: TreeState) -> Literal["expand", "__end__"]:
            """Determine whether to continue the tree search."""
            self.root = state["root"]
            if self.root.is_solved:
                return END
            if self.root.height > 4:
                return END
            return "expand"

        builder = StateGraph(TreeState)
        builder.add_node("start", self.generate_initial_response)
        builder.add_node("expand", self.expand)
        builder.set_entry_point("start")

        builder.add_conditional_edges(
            "start",
            # Either expand/rollout or finish
            should_loop,
        )
        builder.add_conditional_edges(
            "expand",
            # Either continue to rollout or finish
            should_loop,
        )

        graph = builder.compile()
        return graph

    def start(self, **args):
        last_step = None
        for step in self.graph.stream({"input": self.io_func.input_description}):
            last_step = step
            step_name, step_state = next(iter(step.items()))
            print(step_name)
            print("rolled out: ", step_state["root"].height)
            print("---")
        if step_state["root"].height == 1:
            solution_node = last_step["start"]["root"].get_best_solution()
            best_trajectory = solution_node.get_trajectory(include_reflections=False)
            print(best_trajectory[-1].content)
        else:
            solution_node = last_step["expand"]["root"].get_best_solution()
            best_trajectory = solution_node.get_trajectory(include_reflections=False)
            print(best_trajectory[-1].content)

    def to_json(self, answer=True, process=True):

        if process:
            json_obj = {
                "solved": self.root.is_solved == 1,
                "tree": self.root.to_json_recursive(),
            }
        else:
            json_obj = {}
        if answer:
            json_obj["answer_generation"] = {
                # "query_count": self.query_count,
                # "total_tokens": self.total_tokens,
                "final_answer": "",
                "function": self.io_func.functions,
            }
        return json_obj

# with open('env_99.pkl', 'rb') as file:
#     env = pickle.load(file)
# search = LATSSearch(None, env)
# search.start()
# s = search.to_json(True, True)
# print(1)
