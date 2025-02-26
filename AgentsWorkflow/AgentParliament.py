import random
import json
from collections import Counter
from llm_service.llm_generator import generate_llm_response

# --- Discussion Mesh Structure ---
class DiscussionNode:
    def __init__(self, agent_name, prompt, response, level):
        self.agent_name = agent_name
        self.prompt = prompt
        self.response = response
        self.level = level
        self.children = []

def print_discussion_tree(node, indent=0):
    indent_str = "  " * indent
    print(f"{indent_str}{node.agent_name} (Level {node.level}):")
    print(f"{indent_str}  Prompt: {node.prompt}")
    print(f"{indent_str}  Response: {node.response}")
    for child in node.children:
        print_discussion_tree(child, indent + 1)

def get_transcript_string(node, indent=0):
    indent_str = "  " * indent
    transcript = f"{indent_str}{node.agent_name} (Level {node.level}):\n"
    transcript += f"{indent_str}  Prompt: {node.prompt}\n"
    transcript += f"{indent_str}  Response: {node.response}\n"
    for child in node.children:
        transcript += get_transcript_string(child, indent + 1)
    return transcript

# --- Right To Information (RTI) Component ---
class RightToInformation:
    def __init__(self):
        self.discussion_transcript = None
        self.decision_log = []

    def update_discussion(self, discussion_tree):
        self.discussion_transcript = discussion_tree

    def update_decision_log(self, decision_log):
        self.decision_log = decision_log

    def get_transparency_report(self):
        report = "=== Transparency Report ===\n\n"
        if self.discussion_transcript:
            report += "Discussion Transcript:\n"
            report += get_transcript_string(self.discussion_transcript)
        else:
            report += "No discussion transcript available.\n"
        if self.decision_log:
            report += "\nDecision Log:\n"
            for decision in self.decision_log:
                report += f"Issue: {decision['issue']}\n"
                report += f"Proposal: {decision['proposal']}\n"
                report += f"Votes: {decision['votes']}\n"
                report += f"Final Answer: {decision['final_answer']}\n"
                report += f"Most Prevalent Reason: {decision['final_reason']}\n\n"
        else:
            report += "No decision log available.\n"
        return report

# --- Validator Function ---
def validate_solution(aggregated_reasoning, final_solution):
    prompt = (
        "Here is the aggregated chain-of-thought and reasoning:\n" + aggregated_reasoning +
        "\nBased on the above reasoning, the final proposed solution is:\n" + final_solution +
        "\nPlease assess whether the reasoning is consistent and whether the final solution is supported. "
        "Provide a brief validation message."
    )
    validation = generate_llm_response(prompt)
    return validation

# --- Base Agent Class ---
class Agent:
    def __init__(self, name):
        self.name = name

    def discuss(self, prompt, level=1):
        response = f"{self.name} provides initial thoughts on '{prompt}'."
        return DiscussionNode(self.name, prompt, response, level)

# --- Departmental Subagent ---
class DepartmentalSubagent(Agent):
    def analyze_issue(self, issue):
        prompt = f"As {self.name}, analyze the following problem: {issue}"
        analysis = generate_llm_response(prompt)
        print(f"{self.name} analysis: {analysis}")
        return analysis

    def discuss(self, prompt, level):
        analysis = self.analyze_issue(prompt)
        response = f"Detailed insight: {analysis}"
        return DiscussionNode(self.name, prompt, response, level)

# --- Domain Expert Agent ---
class DomainExpertAgent(Agent):
    def discuss(self, prompt, level=1):
        llm_prompt = (
            f"As {self.name}, an expert in this domain, provide a comprehensive analysis and solution approach for the following problem:\n{prompt}"
        )
        response = generate_llm_response(llm_prompt)
        return DiscussionNode(self.name, prompt, response, level)
    
    def vote(self, problem):
        llm_prompt = (
            f"As {self.name}, given the problem:\n{problem}\n"
            "Provide your proposed solution along with a brief explanation."
        )
        proposal = generate_llm_response(llm_prompt)
        return (proposal, f"Expert rationale: {proposal[:100]}...")

# --- Government Agent (Minister) ---
class GovernmentAgent(Agent):
    def __init__(self, name, subagents):
        super().__init__(name)
        self.subagents = subagents
        self.proposal = None

    def prepare_policy(self, problem):
        analyses = [subagent.analyze_issue(problem) for subagent in self.subagents]
        context = " | ".join(analyses)
        llm_prompt = (
            f"As {self.name}, a government agent responsible for solving complex problems, "
            f"consider the following problem:\n{problem}\n\n"
            f"With the following context from preliminary analysis: {context}\n\n"
            "Generate a detailed, step-by-step solution proposal that is general and adaptable."
        )
        self.proposal = generate_llm_response(llm_prompt)
        print(f"{self.name} generated proposal: {self.proposal}")
        return self.proposal

    def vote(self, problem):
        if not self.proposal:
            self.prepare_policy(problem)
        return (self.proposal, f"Government rationale (from {self.name}): {self.proposal[:100]}...")

    def discuss(self, prompt, level=1):
        sub_discussions = [subagent.discuss(prompt, level+1) for subagent in self.subagents]
        combined = " | ".join(d.response for d in sub_discussions)
        response = f"{self.name} synthesizes: {combined}"
        return DiscussionNode(self.name, prompt, response, level)

# --- Opposition Agent ---
class OppositionAgent(Agent):
    def discuss(self, prompt, level=1):
        llm_prompt = (
            f"As {self.name}, an opposition agent, analyze the following problem and point out potential pitfalls or improvements:\n{prompt}"
        )
        response = generate_llm_response(llm_prompt)
        return DiscussionNode(self.name, prompt, response, level)
    
    def vote(self, problem):
        llm_prompt = (
            f"As {self.name}, given the problem:\n{problem}\n"
            "Provide your proposed solution (which may critique the government proposal) along with your rationale."
        )
        proposal = generate_llm_response(llm_prompt)
        return (proposal, f"Opposition rationale: {proposal[:100]}...")

# --- Speaker / Moderator ---
class Speaker(Agent):
    def moderate_session(self, message):
        print(f"\nSpeaker {self.name} moderates: {message}")

    def discuss(self, prompt, level=0):
        response = f"{self.name} sets the overall perspective: {prompt}"
        return DiscussionNode(self.name, prompt, response, level)

# --- Voting Module ---
class VotingModule:
    def __init__(self, voters):
        self.voters = voters

    def conduct_vote(self, problem):
        votes = {}
        print("\n--- Voting Session ---")
        for voter in self.voters:
            vote, reason = voter.vote(problem)
            votes[voter.name] = (vote, reason)
        return votes

    def tally_votes(self, votes):
        proposals = [vote for vote, _ in votes.values()]
        aggregation_prompt = (
            "Given the following solution proposals and their rationales:\n" +
            "\n".join([f"{name}: {vote}" for name, (vote, _) in votes.items()]) +
            "\n\nPlease synthesize a final, coherent solution and explain the most compelling reasoning behind it."
        )
        final_solution = generate_llm_response(aggregation_prompt)
        rationale_prompt = (
            "Based on the following rationales:\n" +
            "\n".join([f"{name}: {reason}" for name, (_, reason) in votes.items()]) +
            "\n\nExtract the most compelling rationale that supports the final solution."
        )
        prevalent_reason = generate_llm_response(rationale_prompt)
        print(f"\nFinal synthesized solution: {final_solution}")
        print(f"Extracted prevailing rationale: {prevalent_reason}")
        return final_solution, prevalent_reason

# --- Parliament Engine ---
class ParliamentEngine:
    def __init__(self, speaker, government_agents, opposition_agents, domain_experts, problem, rti):
        self.speaker = speaker
        self.government_agents = government_agents
        self.opposition_agents = opposition_agents
        self.domain_experts = domain_experts
        self.problem = problem
        self.decision_log = []
        self.discussion_tree = None
        self.rti = rti

    def run_session(self):
        self.speaker.moderate_session(f"Session started for problem: {self.problem}")
        proposals = []
        for agent in self.government_agents:
            proposals.append(agent.prepare_policy(self.problem))
        combined_proposal = " || ".join(proposals)
        print(f"\nCombined Proposal from Government Agents:\n{combined_proposal}")
        print("\n--- Opposition Round ---")
        for opp in self.opposition_agents:
            opp.question_policy = opp.discuss(self.problem, level=1)
        all_voters = self.government_agents + self.opposition_agents + self.domain_experts
        voting_module = VotingModule(all_voters)
        votes = voting_module.conduct_vote(self.problem)
        final_solution, prevalent_reason = voting_module.tally_votes(votes)
        session_log = {
            "issue": self.problem,
            "proposal": combined_proposal,
            "votes": votes,
            "final_answer": final_solution,
            "final_reason": prevalent_reason
        }
        self.decision_log.append(session_log)
        self.rti.update_decision_log(self.decision_log)
        self.speaker.moderate_session("Session ended.")
        return final_solution, prevalent_reason

    def run_hierarchical_discussion(self, prompt):
        top_node = self.speaker.discuss(prompt, level=0)
        for agent in self.government_agents:
            node = agent.discuss(prompt, level=1)
            top_node.children.append(node)
        for expert in self.domain_experts:
            node = expert.discuss(prompt, level=1)
            top_node.children.append(node)
        for opp in self.opposition_agents:
            node = opp.discuss(prompt, level=1)
            top_node.children.append(node)
        self.discussion_tree = top_node
        self.rti.update_discussion(self.discussion_tree)
        self.rti.update_decision_log(self.decision_log)
        print("\nDiscussion Mesh:")
        print_discussion_tree(top_node)

    def get_llm_summary(self, prompt="Please summarize the discussion transcript"):
        transcript = get_transcript_string(self.discussion_tree) if self.discussion_tree else "No transcript available."
        full_prompt = f"{prompt}\n\nDiscussion Transcript:\n{transcript}"
        summary = generate_llm_response(full_prompt)
        return summary

# --- Dynamic Role Determination ---
def determine_roles(problem_text):
    prompt = (
        "Based on the following problem, provide a JSON specification of the agent roles required to solve it. "
        "Include the keys: 'government_agents', 'domain_experts', 'opposition_agents', and 'subagents_per_government'.\n\n"
        f"Problem: {problem_text}\n\n"
        "Return a valid JSON object."
    )
    response = generate_llm_response(prompt)
    try:
        spec = json.loads(response)
        keys = ["government_agents", "domain_experts", "opposition_agents", "subagents_per_government"]
        if all(key in spec for key in keys):
            return spec
    except Exception as e:
        print("Error parsing role specification from LLM:", e)
    return {"government_agents": 2, "domain_experts": 1, "opposition_agents": 2, "subagents_per_government": 2}

