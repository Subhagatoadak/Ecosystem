from AgentsFramework.AgentParliament.AgentParliament import *

# --- Main Function ---
def main():
    print("Welcome to the Dynamic Multi-Agent Problem Solving Framework.\n")
    problem = input("Please describe the complex problem you want to solve:\n").strip()
    if not problem:
        problem = (
            "You have 10 bags full of coins. Each bag has infinite coins. One bag contains forged coins weighing 1.1 g each, "
            "while genuine coins weigh 1 g. Using a digital weighing machine and taking a different number of coins from each bag, "
            "determine which bag contains the forged coins with minimal readings."
        )
    print(f"\nProcessing problem:\n{problem}\n")
    
    manual_flag = input("Do you want to manually specify agent roles? (y/n): ").strip().lower()
    if manual_flag == 'y':
        role_spec = {}
        try:
            role_spec["government_agents"] = int(input("Enter number of government agents: "))
            role_spec["domain_experts"] = int(input("Enter number of domain experts: "))
            role_spec["opposition_agents"] = int(input("Enter number of opposition agents: "))
            role_spec["subagents_per_government"] = int(input("Enter number of subagents per government agent: "))
        except Exception as e:
            print("Invalid input, falling back to dynamic role determination.")
            role_spec = determine_roles(problem)
    else:
        role_spec = determine_roles(problem)
    print("Role specification:", role_spec)
    
    gov_agents = []
    for i in range(role_spec["government_agents"]):
        subs = [DepartmentalSubagent(f"Gov{i+1} Subagent {j+1}") for j in range(role_spec["subagents_per_government"])]
        gov_agents.append(GovernmentAgent(f"Minister {i+1}", subs))
    
    domain_experts = [DomainExpertAgent(f"Domain Expert {i+1}") for i in range(role_spec["domain_experts"])]
    opposition_agents = [OppositionAgent(f"Opposition {i+1}") for i in range(role_spec["opposition_agents"])]
    speaker = Speaker("Moderator")
    
    rti = RightToInformation()
    
    parliament = ParliamentEngine(
        speaker,
        gov_agents,
        opposition_agents,
        domain_experts,
        problem,
        rti
    )
    
    # For demonstration: if the problem involves coins and bags, simulate a digital weighing reading.
    if "coin" in problem.lower() and "bag" in problem.lower():
        counterfeit_bag = random.randint(1, 10)
        measured_weight = 55 + 0.1 * counterfeit_bag
        print(f"\nSimulated digital weighing reading: {measured_weight:.1f} grams.")
    else:
        measured_weight = None
    
    if measured_weight is not None:
        final_solution, prevalent_reason = parliament.run_session()
        print("\n=== Final Synthesized Solution ===")
        print(final_solution)
        print("\nMost compelling rationale:")
        print(prevalent_reason)
        
        vote_dict = rti.decision_log[-1]["votes"] if rti.decision_log else {}
        aggregated_reasoning = " ".join([reason for _, reason in vote_dict.values()])
        validation = validate_solution(aggregated_reasoning, final_solution)
        print("\n=== Validation of Final Solution ===")
        print(validation)
    else:
        print("No measurable value required for this problem; proceeding with reasoning only.")
        final_solution, prevalent_reason = parliament.run_session()
    
    parliament.run_hierarchical_discussion(
        f"Discuss in detail the steps and reasoning to solve the following problem:\n{problem}\n"
        "Explain how the solution was derived and why each agent role contributed to the final answer."
    )
    
    llm_summary = parliament.get_llm_summary("Please summarize the comprehensive discussion on solving the problem:")
    print("\n=== LLM Generated Summary ===")
    print(llm_summary)
    
    transparency_report = rti.get_transparency_report()
    print("\n=== Full Transparency Report ===")
    print(transparency_report)
    
    # Finally, print the final answer clearly.
    print("\n=== FINAL ANSWER ===")
    print("The final answer to the problem is:")
    print(final_solution)

if __name__ == "__main__":
    main()
