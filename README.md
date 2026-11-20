# SAFIRE

This repository is about the research SAFIRE: Sentiment as a Framing Instrument for Revealing
Exploits in Language Models. 

Abstract

LLMs are increasingly being deployed in high-stakes scenarios where safety and alignment with hu
man values are critical. Despite the use of content moderation and alignment techniques, these models
remain vulnerable to jailbreak attacks, where carefully crafted prompts can bypass safety constraints.
While prior work has primarily focused on structural prompt manipulation and role-based attacks, the in
fluence of sentiment on jailbreak success remains largely underexplored. This work investigates whether
and how sentiment-driven prompt variations affect the ability to elicit harmful responses from general
purpose LLMs. By using previous datasets’ prompts and creating sentimental framings, we created
SAFIRE, a benchmark that pairs 578 harmful requests from five harm categories with five positive and
five negative framings of increasing intensity, written in three independent variation sets (19,074 prompts
in total) and based on politeness and impoliteness theory. This benchmark enables controlled and repro
ducible evaluation of sentiment as an attack. Using this framework, we evaluated LLMs under identical
jailbreak scenarios, analyzed compliance rates and compared sentiment-based attacks with their neu
tral baselines. We reached the conclusion that the effectiveness of sentiment-driven prompting for this
task depends on the model that is being attacked. Of the six victim models that were tested, four of
them produced more compliant responses with neutral prompts than with sentimental ones. However,
for Llama 3 and 4 positive intensities worked best. This leads us to the conclusion that these results
depend on the victim model.

The repository is organized as follows:

    \Experiment 1 - Sentiment Analysis
        
        \Complete Datasets
            Has datasets SST-3 and DynaSent
        
        \Results
            \DynaSent
                Results for DynaSent

            \SST-3
                Results for SST-3

        \Splits
            Has the script we used to create the test and train splits for both datasets


    \Experiment 2 - Judge Ensemble selection
        
        \Dataset Sent to Victim LLMs
            Contains the dataset used for Experiment 2 as well as the scripts we used to send it to the victim LLMs

        \Experiment Results
            Contains the compliant responses of the victims as well as the scripts used for the experiment

            \output_9
                Results for the experiment with 9 judges

            \output_11
                Results for the experiment with 11 judges

        \Judge Ensemble Results
            Results analyzing the chosen ensemble for these results

        \Judge Results
            Results for each of the 11 judges

        \Judge Scripts
            Scripts used to judge the jailbreak results

        \Victim LLMs Answers
            Victim LLM responses to the dataset (jailbreak results)


    \Experiment 3 - Jailbreaks Compliance
        
        \Experiment 3.1 - Neutral Prompt Variation
            Has the script used to run experiment 3.1 and its results

        \Experiment 3.2 - Judge Ensemble (Human Annotator)
            Scripts and results for Experiment 3.2

        \Judge Scripts
            Scripts used to judge the jailbreak results

        \Scripts to Send Prompt to Victim
            Scripts used to send harmful prompts to the victim LLMs

        \Statistical Analysis
            Script for statistical analysis and its results


    \SAFIRE Dataset
        Contains the SAFIRE dataset, as well as the jailbreak and judged results for Experiment 3


    \Script for Experiment 3 Pipeline
        Contains a script that performs all steps for experiment 3 (except human validation)

