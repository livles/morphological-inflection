TRAIN_FILEPATH = ""
TEST_FILEPATH = ""
LANGS = []


def preprocess(x):
    lemma, features, target = x.split("\t")
    prompt = "inflect " + lemma + " using " + features
    target_string = lemma+"\t"+features+"\t"+target
    


for lang in LANGS:
    with open(TRAIN_FILEPATH,"r") as train_file, open(TEST_FILEPATH,"r") as test_file:
        train_lines = train_file.readlines()
        test_lines = test_file.readlines()
        train_prompts = map(f,train_lines)
        test_prompts = map(f,test_lines)