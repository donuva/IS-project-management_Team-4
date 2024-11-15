import argparse
import transformers
from transformers import AutoTokenizer
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from datasets import load_dataset
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from bert_score import score

# Define a function to generate summaries and populate the 'model_generated' column
def generate_and_store_summary(row):
  article_text = row['article']
  summary = llm_chain.run(article_text)
  return summary

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Summarization using mpt_7b_instruct on CNN_Dailymail.")
  parser.add_argument("-n", "--num_summaries", type=int, default=-1, help="Number of summaries (-1 for all)")  
  parser.add_argument("-o", "--output_file", type=str, default="summaries.csv", help="Path to save the CSV file (default: summaries.csv)") 

  args = parser.parse_args()

  model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True
  )
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

  pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100, do_sample=True, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
  llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0.1})

  # Using LangChain and HuggingFacePipeline for Prompting of text summarization
  template = """
                Write a concise summary of the following text delimited by triple backquotes.
                ```{text}```
                SUMMARY:
            """

  prompt = PromptTemplate(template=template, input_variables=["text"])

  llm_chain = LLMChain(prompt=prompt, llm=llm)

  # load dataset 
  dataset = load_dataset('cnn_dailymail', '3.0.0')
  test_df = pd.DataFrame(dataset['test'])

  test_df.drop(columns=['id'], inplace=True)

  # Create an empty column 'model_generated' in test_df to store the generated summaries
  test_df['model_generated'] = ""

  if args.num_summaries == -1:
      # Process the entire DataFrame
      num_records_to_process = len(test_df)  
  else:
      # Process the specified number of records
      num_records_to_process = args.num_summaries  

  test_df.loc[:num_records_to_process - 1, 'model_generated'] = test_df.loc[:num_records_to_process - 1].apply(generate_and_store_summary, axis=1)
  print("Finished generating summaries for required number of articles.")
  # Save the DataFrame to a CSV file if the output_file argument is provided
  if args.output_file:
    test_df.to_csv(args.output_file, index=False)  
    print(f"Summaries saved to: {args.output_file}")

  #################################
  # EVALUATION
  # Initialize the ROUGE evaluator
  rouge = Rouge()

  # Select the number of records
  sampled_df = test_df.head(num_records_to_process)

  # Extract the generated summaries and reference summaries for the selected samples
  generated_summaries = sampled_df['model_generated'].tolist()
  reference_summaries = sampled_df['highlights'].tolist()

  # Calculate ROUGE scores for the selected samples
  rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

  # Print the ROUGE scores
  print("ROUGE Scores:", rouge_scores)

  # Calculate BLEU score for the selected samples
  bleu_score = corpus_bleu(reference_summaries, generated_summaries)
  print("BLEU Score:", bleu_score)

  # Calculate BERT Score
  P, R, F1 = score(generated_summaries, reference_summaries, lang="en", verbose=True)

  # Print BERT Score
  print("BERT Precision:", P.mean().item())
  print("BERT Recall:", R.mean().item())
  print("BERT F1 Score:", F1.mean().item())

