SETTINGS = ["grc","dan","fra","sme","deu","nav","jap","klr","eng","mul","deu_eng"] + ["mul_2","ote","pol","amh"]
SETTINGS = SETTINGS[-4:]
TEST_LANGS_OLD = ["grc","dan","fra","sme","deu","nav","jap","klr","eng"] 
TEST_LANGS = ["grc","dan","fra","sme","deu","nav","jap","klr","eng"] + ["ote","pol","amh","csb"]
# TEST_LANGS = []
for lang_setting in SETTINGS:

    if lang_setting in ( "deu_eng", "mul"): test_langs = TEST_LANGS_OLD
    elif lang_setting == "mul_2": test_langs = TEST_LANGS
    elif lang_setting == "pol": test_langs = ["pol","csb"]
    else:
        test_langs = [lang_setting]
    for test_lang in test_langs:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        PRETRAINED_MODEL = "google/byt5-small"
        FINETUNED_MODEL ="livles/"+lang_setting
        FINETUNED_MODEL ="./byt5_small/"+lang_setting
        MODEL = FINETUNED_MODEL
        print(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            FINETUNED_MODEL,
        dtype=torch.float16,
        device_map="auto"
        )

        from datasets import load_dataset
        OUT_DIR = "output_byt5_small/" + lang_setting + "/" 
        PATH_TO_JSON_TEST_FILE = "../preprocessing_to_json/data/" + test_lang + "_test.json"
        PATH_TO_ORIG_TEST_FILE = "../2023InflectionST/part1/data/" + test_lang + ".tst"
        LEN = 1000
        with open (OUT_DIR + test_lang + "_byt5-small.out","w") as out_file, open (PATH_TO_ORIG_TEST_FILE, "r") as test_file:
            COUNTER = 0
            dataset = load_dataset("json",data_files={"test":PATH_TO_JSON_TEST_FILE})
            covered_test_lines = dataset["test"]["input"]


            # preds = map (generate,covered_test_lines)
            for line in covered_test_lines:
                input_ids = tokenizer(line, return_tensors="pt").to(model.device)
                output = model.generate(
                    **input_ids,
                    max_new_tokens=50,
                    num_beams=4,
                    #early_stopping=True
                )
                output_string = (tokenizer.decode(output[0], skip_special_tokens=True))
                lemma, features, ref = test_file.readline().strip().split("\t")

                # evaluate
                if (ref == output_string):
                    COUNTER += 1
                    print(ref,output_string)
                else:
                    print("ref:",ref,"pred:",output_string)
                
                # write
                out_file.write(lemma + "\t" +features+ "\t" + output_string + "\n")
                out_file.flush()

            accuracy = COUNTER / 1000
            out_file.write("accuracy:"+ str(accuracy))
            print(accuracy)
