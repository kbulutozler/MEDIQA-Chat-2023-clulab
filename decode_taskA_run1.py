import pandas as pd
import numpy as np
import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import torch


def arguments():
    parser = argparse.ArgumentParser(description='arguments for the decoding.')

    parser.add_argument("-t", "--test_filepath", help="Get the test filepath.", default="test", type=str)
    args=parser.parse_args()
    return args

def main():
    args=arguments()
    summary_model_path = "./summary_model"
    summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_path)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_path)
    header_model_path = "./header_model"
    header_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    header_model = AutoModelForSequenceClassification.from_pretrained(header_model_path)
    labels = ['GENHX', 'MEDICATIONS', 'CC', 'PASTMEDICALHX', 'ALLERGY', 'FAM/SOCHX', 'PASTSURGICAL', 'OTHER_HISTORY', 'ASSESSMENT', 'ROS', 'DISPOSITION', 'EXAM', 'PLAN', 'DIAGNOSIS', 'EDCOURSE', 'IMMUNIZATIONS', 'LABS', 'IMAGING', 'PROCEDURES', 'GYNHX']
    idx2label = {i:l for i,l in enumerate(labels)}
    df_test = pd.read_csv(args.test_filepath)
    # "TestID", "SystemOutput1", and "SystemOutput2"
    submission = {"TestID":[], "SystemOutput1":[], "SystemOutput2":[]}
    for i, row in df_test.iterrows():
        print(i)
        dialogue = row["dialogue"]
        test_id = row["ID"]
        submission["TestID"].append(test_id)

        
        summary_inputs = summary_tokenizer(dialogue, return_tensors="pt").input_ids
        summary_outputs = summary_model.generate(summary_inputs, max_new_tokens=100, do_sample=False)
        gen_summary = summary_tokenizer.decode(summary_outputs[0], skip_special_tokens=True)
        submission["SystemOutput2"].append(gen_summary)
        
        header_inputs = header_tokenizer(gen_summary, return_tensors="pt")
        with torch.no_grad():
            logits = header_model(**header_inputs).logits
        predicted_class_id = logits.argmax().item()
        label = idx2label[predicted_class_id]
        submission["SystemOutput1"].append(label)
        
    submission = pd.DataFrame.from_dict(submission)
    submission.to_csv("./taskA_clulab_run1.csv", index=False)


if __name__ == '__main__':
    main()