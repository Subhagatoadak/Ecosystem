import random

# Base class for all agents
class Agent:
    def __init__(self, name):
        self.name = name

# DepartmentalSubagent breaks down issues within a ministry
class DepartmentalSubagent(Agent):
    def analyze_issue(self, issue):
        analysis = f"{self.name} analysis on '{issue}'"
        print(f"{self.name} analyzing issue: {issue}")
        return analysis

# GovernmentAgent (Minister) collects subagent input and formulates a policy proposal.
class GovernmentAgent(Agent):
    def __init__(self, name, subagents):
        super().__init__(name)
        self.subagents = subagents
        self.proposal = None

    def prepare_policy(self, issue):
        print(f"\n{self.name} is preparing policy for issue: {issue}")
        # Gather detailed analysis from subagents
        analysis_reports = []
        for subagent in self.subagents:
            analysis = subagent.analyze_issue(issue)
            analysis_reports.append(analysis)
        # Combine subagent analysis into a policy proposal
        self.proposal = f"Policy by {self.name}: " + " | ".join(analysis_reports)
        print(f"{self.name} prepared proposal: {self.proposal}")
        return self.proposal

    def vote(self, proposal):
        # In this simulation, vote randomly (could be based on further logic)
        vote = random.choice(["Yes", "No"])
        print(f"{self.name} votes {vote}")
        return vote

# OppositionAgent questions and challenges the proposals
class OppositionAgent(Agent):
    def question_policy(self, proposal):
        question = f"{self.name} questions the proposal: {proposal}"
        print(question)
        return question

# Speaker (Moderator) controls the session and enforces protocol
class Speaker(Agent):
    def moderate_session(self, message):
        print(f"\nSpeaker {self.name} moderates: {message}")

# VotingModule collects votes from government agents and determines the outcome.
class VotingModule:
    def __init__(self, government_agents):
        self.government_agents = government_agents

    def conduct_vote(self, proposal):
        votes = {}
        print("\n--- Voting Session ---")
        for agent in self.government_agents:
            vote = agent.vote(proposal)
            votes[agent.name] = vote
        return votes

    def tally_votes(self, votes):
        yes_votes = sum(1 for vote in votes.values() if vote == "Yes")
        no_votes = sum(1 for vote in votes.values() if vote == "No")
        decision = "Accepted" if yes_votes > no_votes else "Rejected"
        print(f"\nVoting result: {yes_votes} Yes, {no_votes} No. Proposal is {decision}.")
        return decision

# ParliamentEngine coordinates the overall session
class ParliamentEngine:
    def __init__(self, speaker, government_agents, opposition_agents, issue):
        self.speaker = speaker
        self.government_agents = government_agents
        self.opposition_agents = opposition_agents
        self.issue = issue
        self.decision_log = []

    def run_session(self):
        # Session initiation
        self.speaker.moderate_session(f"Session started for issue: {self.issue}")

        # Government agents prepare their policy proposals using subagents
        proposals = []
        for agent in self.government_agents:
            proposal = agent.prepare_policy(self.issue)
            proposals.append(proposal)
        
        # For simplicity, the system combines proposals into one unified proposal.
        combined_proposal = "Combined Proposal: " + " || ".join(proposals)
        print(f"\n{combined_proposal}")

        # Opposition agents challenge the combined proposal.
        print("\n--- Opposition Round ---")
        for opp in self.opposition_agents:
            opp.question_policy(combined_proposal)

        # Conduct vote among government agents.
        voting_module = VotingModule(self.government_agents)
        votes = voting_module.conduct_vote(combined_proposal)
        decision = voting_module.tally_votes(votes)

        # Log the decision for record keeping.
        self.decision_log.append({
            "issue": self.issue,
            "proposal": combined_proposal,
            "votes": votes,
            "decision": decision
        })
        self.speaker.moderate_session("Session ended.")
        print("\n--- Decision Log ---")
        for log in self.decision_log:
            print(log)

def main():
    # Create departmental subagents for each government agent
    subagents1 = [DepartmentalSubagent("Subagent1.1"), DepartmentalSubagent("Subagent1.2")]
    subagents2 = [DepartmentalSubagent("Subagent2.1"), DepartmentalSubagent("Subagent2.2")]
    subagents3 = [DepartmentalSubagent("Subagent3.1"), DepartmentalSubagent("Subagent3.2")]

    # Create government agents (ministers) with their respective subagents
    gov_agent1 = GovernmentAgent("Minister1", subagents1)
    gov_agent2 = GovernmentAgent("Minister2", subagents2)
    gov_agent3 = GovernmentAgent("Minister3", subagents3)

    # Create opposition agents
    opp_agent1 = OppositionAgent("Opposition1")
    opp_agent2 = OppositionAgent("Opposition2")

    # Create the speaker (moderator)
    speaker = Speaker("Speaker1")

    # Define an issue for discussion
    issue = "Increase in renewable energy investment"

    # Initialize the Parliament Engine with all agents and the issue
    parliament = ParliamentEngine(
        speaker,
        [gov_agent1, gov_agent2, gov_agent3],
        [opp_agent1, opp_agent2],
        issue
    )
    
    # Run the parliamentary session
    parliament.run_session()

if __name__ == "__main__":
    main()
