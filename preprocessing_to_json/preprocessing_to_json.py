TRAIN_FILEPATH = ""
TEST_FILEPATH = ""
OUTPUT_TRAIN_FILEPATH = ""
OUTPUT_TEST_FILEPATH = ""  
LANGS = []


def preprocess(x):
    lemma, features, target = x.split("\t")
    prompt = "inflect " + lemma + " using " + features
    target_string = lemma+"\t"+features+"\t"+target
    line = '{"text":"'+prompt+'","summary":"'+target_string+'"}'


for lang in LANGS:
    with open(TRAIN_FILEPATH,"r") as train_file, open(TEST_FILEPATH,"r") as test_file, open (OUTPUT_TRAIN_FILEPATH + "_"+lang+ ".json","w") out_train_file, open (OUTPUT_TEST_FILEPATH + "_" + lang + ".json", "w") as out_test_file:
        train_lines = train_file.readlines()
        test_lines = test_file.readlines()
        train_lines_list = map(f,train_lines)
        test_json_lines_list = map(f,test_lines)
        train_json_lines = "\n".join(train_lines_list)
        test_json_lines = "\n".join(test_json_lines_list)

