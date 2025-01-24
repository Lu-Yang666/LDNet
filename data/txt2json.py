import os
import json
import sys
import random
import re
from collections import defaultdict

rel_inst = ['How do two words relate to each other based on the sentence that describes them?',
            'Identify two words and how they are related based on a phrase that explains their connection.',
            'Given a sentence, please extract the subject and object containing a certain relation in the sentence.',
            'What is the connection between two words according to the sentence that explains them?',
            'How are two words related based on the sentence that describes them?',
            'Identify the subject and object that have a specific relation in a sentence.What is the relationship between two words according to the given sentence?',
            'Find the phrases in the following sentences that have a given relationship.',
            'Given a phrase that describes the relationship between two words, extract the words and the lexical relationship between them.',
            'Locate the parts of the sentences that are connected by a specific relation.']
rel = ["None", "/per/per/parent", "/per/per/siblings", "/per/per/couple", "/per/per/neighbor", "/per/per/peer",
       "/per/per/charges", "/per/per/alumi", "/per/org/member_of",
       "/per/loc/place_of_residence", "/per/loc/place_of_birth", "/org/org/alternate_names", "/org/org/subsidiary",
       "/org/loc/locate_at", "/loc/loc/contain", "/per/misc/present_in", "/per/misc/awarded", "/per/misc/race",
       "/per/misc/religion", "/per/misc/nationality", "/misc/misc/part_of", "/misc/loc/held_on"]
rel = [relation.split("/")[-1].replace("_", " ") for relation in rel]
ent_inst = [
    'Would you be able to recognize any entities that may be present in the provided text and label them based on their types?',
    'Could you recognize any possible entities from the provided text and determine their types?',
    'Would you be able to identify any entities in the provided text and label them based on their respective types?',
    'Please recognize any entities that might exist in the given text and classify them according to their respective types.',
    'Can you detect any possible entities from the given text and determine their corresponding types?',
    'Please recognize any possible entities in the given text and determine their respective types.',
    'Can you detect any entities that may be present in the given text and classify them based on their types?',
    'Would you be able to detect any potential entities in the given text and categorize them based on their types?',
    'Could you detect any possible entities from the given text and label them based on their types?',
    'Can you identify any potential entities in the provided text and categorize them based on their types?',
    'Can you pinpoint any entities that might exist in the provided text and label them based on their types?',
    'From the provided text, could you identify any entities and determine their types?',
    'Please identify any potential entities in the given text and determine their corresponding types.',
    'Please identify any entities in the provided text and classify them based on their types.',
    'Could you recognize any entities that may be present in the provided text and classify them according to their types?',
    'Please recognize any possible entities in the given text and categorize them based on their types.',
    'From the given text, could you pinpoint any possible entities and determine their types?',
    'Can you detect possible entities in the given text and categorize them based on their types?',
    'From the provided text, could you detect any potential entities and classify them according to their respective types?',
    'Please identify any possible entities from the given text and determine their corresponding types.',
    'Would you be able to identify any potential entities in the given text and determine their corresponding types?',
    'Please identify any entities that could be present in the given text and classify them according to their types.',
    'Could you pinpoint any potential entities that may exist in the given text and label them based on their respective types?',
    'Can you detect any entities that could be present in the given text and categorize them based on their types?',
    'Can you identify any entities that could be present in the provided text and classify them according to their types?',
    'Please identify any possible entities in the given text and label them based on their types.',
    'Can you pinpoint any potential entities in the given text and label them based on their respective types?',
    'Could you recognize entities that might exist in the given text and classify them according to their types?',
    'Could you identify any entities that might exist in the given text and categorize them based on their respective types?',
    'From the given text, could you identify any entities and determine their respective types?',
    'Would you be able to recognize any possible entities in the given text and determine their corresponding types?',
    'Please identify any entities that might exist in the given text and classify them according to their respective types.',
    'Please identify possible entities from the given text and determine their types.',
    'Please identify any potential entities in the given text and categorize them according to their types.',
    'Can you detect any possible entities from the given text and classify them according to their types?',
    'From the given text, can you pinpoint any entities and label them based on their types?',
    'Would you be able to recognize any entities that might exist in the provided text and categorize them based on their types?',
    'Would you be able to recognize any possible entities in the given text and determine their types?',
    'From the provided text, could you pinpoint any potential entities and label them based on their types?',
    'From the provided text, could you identify any entities and classify them based on their respective types?',
    'Can you detect any entities in the provided text and label them based on their types?',
    'Please identify any entities in the given text and determine their corresponding types.']
ent = ["per", "loc", "org", "misc"]


class Pre_Process:
    def __init__(self, path_in, path_out, option):
        self.path_in = path_in
        self.path_out = path_out
        self.none = option[0]  # deal with the relation of None
        self.hyper_relation = option[1]  # merge the json with the same text
        self.instruct = option[2]

    def process(self):
        self.ner_preprocess()
        self.re_preprocess()
        self.merged()

    def re_preprocess(self):
        path = 'RE/data/txt'
        for dir in os.listdir(self.path_in + path):
            if dir.endswith("train.txt") or dir.endswith("val.txt") or dir.endswith("test.txt"):
                # 指定新的 JSON 文件路径和文件名
                output_file = self.path_out + path.replace("txt", "mre") + "/" + \
                              "origin_" + dir.replace("txt", "jsonl").split("_")[1].replace("val", "dev")
                output_file_final = self.path_out + path.replace("txt", "mre") + "/" + \
                                    dir.replace("txt", "jsonl").split("_")[1].replace("val", "dev")
                output_dir = os.path.dirname(output_file)
                # print(output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                print(f"write to {output_file}")
                with open(output_file, "w", encoding="utf-8") as outfile:
                    with open(self.path_in + path + "/" + dir, "r", encoding="utf-8") as file:
                        # print(dir)
                        lines = file.readlines()
                        # convert txt to json according to Mirror
                        for i, line in enumerate(lines):
                            line = self.replace(line)
                            # print(f"{i}:{line}")
                            old_line = json.loads(line, strict=False)
                            if old_line["relation"].split("/")[-1].replace("_",
                                                                           " ") == "None" and self.none.upper() == "F":
                                continue

                            text = ' '.join(old_line["token"])
                            # text = text.replace(f" {old_line['h']['name']} ", f" [HEAD]{old_line['h']['name']}[/HEAD] ",1)
                            # text = text.replace(f" {old_line['t']['name']} ", f" [TAIL]{old_line['t']['name']}[/TAIL] ",1)
                            new_line = json.loads("{}")
                            new_line["id"] = f"RE.{dir}.{i}"
                            new_line["schema"] = {
                                "ent": ent,
                                "rel": rel,
                                "event": {}
                            }
                            new_line["ans"] = {
                                # "ent": [
                                #     {
                                #         "type": old_line["relation"].split("/")[1] if old_line[
                                #                                                           "relation"] != "None" else "misc",
                                #         "text": old_line["h"]["name"],
                                #         "span": [text.find(old_line["h"]["name"]),
                                #                  text.find(old_line["h"]["name"]) + len(old_line["h"]["name"])]
                                #     },
                                #     {
                                #         "type": old_line["relation"].split("/")[2] if old_line[
                                #                                                           "relation"] != "None" else "misc",
                                #         "text": old_line["t"]["name"],
                                #         "span": [text.find(old_line["t"]["name"]),
                                #                  text.find(old_line["t"]["name"]) + len(old_line["t"]["name"])]
                                #     }
                                # ],
                                "rel": [
                                    {
                                        "relation": old_line["relation"].split("/")[-1].replace("_", " ") if old_line[
                                                                                                                 "relation"] != "None" else "None",
                                        "head": {
                                            "text": old_line["h"]["name"],
                                            "span": [text.find(old_line["h"]["name"]),
                                                     text.find(old_line["h"]["name"]) + len(old_line["h"]["name"])]
                                        },
                                        "tail": {
                                            "text": old_line["t"]["name"],
                                            "span": [text.find(old_line["t"]["name"]),
                                                     text.find(old_line["t"]["name"]) + len(old_line["t"]["name"])]
                                        }
                                    }
                                ],
                                "event": []
                            }
                            new_line["text"] = text
                            new_line["bg"] = " "
                            new_line["img_id"] = old_line["img_id"]
                            new_line["img_id_dif"] = i
                            new_line["instruction"] = random.choice(rel_inst)
                            if self.instruct.upper() == "T":
                                new_line[
                                    "instruction"] += f"The subject is {old_line['h']['name']} and the object is {old_line['t']['name']}. "
                            # print(new_line)
                            # 将 new_line 写入 JSON 文件
                            json_str = json.dumps(new_line)
                            outfile.write(json_str + '\n')
                            # json.dump(new_line, outfile, ensure_ascii=False, indent=4)

                # text_entitis_dict = defaultdict()  # 存储文本实体键值对
                # current_object = None  # 存储当前正在处理的JSON对象
                # with open(output_file, "r", encoding="utf-8") as infile:
                #     for line in infile:
                #         json_object = json.loads(line)
                #         if current_object == None:
                #             current_object = json_object
                #         elif current_object["text"] == json_object["text"]:
                #             current_object["ans"]["ent"].extend(json_object["ans"]["ent"])
                #         else:
                #             merged_ent = []
                #             for entity in current_object["ans"]["ent"]:
                #                 found_duplicate = False
                #                 for merged in merged_ent:
                #                     if merged["text"] == entity["text"]:
                #                         if merged["type"] == "misc" and entity["type"] != "misc":
                #                             merged["type"] = entity["type"]
                #                         found_duplicate = True
                #                         break
                #                 if not found_duplicate:
                #                     merged_ent.append(entity)
                #
                #             text_entitis_dict[current_object["text"]] = merged_ent
                #             current_object = json_object
                #     # 处理最后一个JSON对象
                #     if current_object is not None:
                #         text_entitis_dict[current_object["text"]] = current_object["ans"]["ent"]
                #
                # refined_lines=[]
                # with open(output_file, "r", encoding="utf-8") as infile:
                #     for line in infile:
                #         json_object = json.loads(line)
                #         for entity in text_entitis_dict[json_object["text"]]:
                #             if entity["text"] == json_object["ans"]["ent"][0]["text"]:
                #                 json_object["ans"]["ent"][0]["type"] = entity["type"]
                #             elif entity["text"] == json_object["ans"]["ent"][1]["text"]:
                #                 json_object["ans"]["ent"][1]["type"] = entity["type"]
                #         refined_lines.append(json_object)
                #
                # with open(output_file, "w", encoding="utf-8") as outfile:
                #     for obj in refined_lines:
                #         json_str = json.dumps(obj)
                #         outfile.write(json_str + '\n')

                if self.hyper_relation.upper() == "T":
                    merged_objects = []  # 存储合并后的JSON对象
                    current_object = None  # 存储当前正在处理的JSON对象
                    with open(output_file, "r", encoding="utf-8") as infile:
                        for line in infile:
                            json_object = json.loads(line)
                            if current_object == None:
                                current_object = json_object
                            elif current_object["text"] == json_object["text"]:
                                current_object["ans"]["ent"].extend(json_object["ans"]["ent"])
                                current_object["ans"]["rel"].extend(json_object["ans"]["rel"])
                            else:
                                merged_ent = []
                                for entity in current_object["ans"]["ent"]:
                                    found_duplicate = False
                                    for merged in merged_ent:
                                        if merged["text"] == entity["text"]:
                                            if merged["type"] == "misc" and entity["type"] != "misc":
                                                merged["type"] = entity["type"]
                                            found_duplicate = True
                                            break
                                    if not found_duplicate:
                                        merged_ent.append(entity)
                                current_object["ans"]["ent"] = merged_ent
                                # ent_mapping = {ent["text"]: ent["type"] for ent in current_object["ans"]["ent"]}
                                # for relation in current_object["ans"]["rel"]:
                                #     relation["head"]["type"] = ent_mapping[relation["head"]["text"]]
                                #     relation["tail"]["type"] = ent_mapping[relation["tail"]["text"]]

                                merged_objects.append(current_object)
                                current_object = json_object
                        # 处理最后一个JSON对象
                        if current_object is not None:
                            merged_objects.append(current_object)

                    with open(output_file_final, "w", encoding="utf-8") as outfile:
                        for obj in merged_objects:
                            json_str = json.dumps(obj)
                            outfile.write(json_str + '\n')
                else:
                    os.rename(output_file, output_file.replace("origin_", ""))

    def ner_preprocess(self):
        path = 'NER/data/'
        for dataset in ["twitter2015", "twitter2017"]:
            nowpath = path + dataset
            for dir in os.listdir(self.path_in + nowpath):
                if dir.endswith("train.txt") or dir.endswith("val.txt") or dir.endswith("test.txt"):
                    # 指定新的 JSON 文件路径和文件名
                    output_file = self.path_out + nowpath + "/" + dir.replace("txt", "jsonl").replace("val", "dev")
                    output_dir = os.path.dirname(output_file)
                    # print(output_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(f"write to {output_file}")
                    with open(output_file, "w", encoding="utf-8") as outfile:
                        with open(self.path_in + nowpath + "/" + dir, "r", encoding="utf-8") as file:
                            # print(dir)
                            lines = file.readlines()
                            count = 0
                            text, labels = [], []
                            for line in lines:
                                # print(line)
                                # image
                                if line.startswith("IMGID:"):
                                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                                    img_id_dif = line.strip() + '.png'
                                    continue
                                # text
                                if line != "\n":
                                    pre_line = line.strip().split('\t')[0]
                                    pre_line = re.sub(' +', '', pre_line)
                                    # print(pre_line)
                                    text.append(pre_line)
                                    label = line.split('\t')[1][:-1]
                                    if 'OTHER' in label:
                                        label = label[:2] + 'MISC'
                                    labels.append(label)
                                else:
                                    # convert
                                    entities = json.loads('[]')
                                    i = 0
                                    pos = 0
                                    while i < len(labels):   #之前的for循环不对，内部i自增没有生效,修复bug
                                        if labels[i] == "O":
                                            pos += len(text[i]) + 1
                                            i += 1
                                            continue
                                        else:
                                            # convert labels to entity
                                            name = [text[i]]
                                            type = labels[i][2:]

                                            while (i < len(labels) - 1 and labels[i + 1] == "I-" + type):
                                                pos += len(text[i]) + 1
                                                i += 1
                                                name.append(text[i])

                                            # print(text)
                                            pos += len(text[i]) + 1
                                            i += 1
                                            text1 = ' '.join(text)
                                            name = ' '.join(name)
                                            start = pos-len(name)-1

                                            entity = {
                                                "type": type.lower(),
                                                "text": name,
                                                "span": [start, start + len(name)]
                                            }
                                            entities.append(entity)


                                    new_line = json.loads("{}")
                                    new_line["id"] = f"NER.{dataset}.{dir}.{count}"
                                    count += 1
                                    new_line["instruction"] = random.choice(ent_inst)
                                    new_line["schema"] = {
                                        "ent": ent,
                                        "rel": rel,
                                        "event": {}
                                    }
                                    new_line["ans"] = {
                                        "ent": entities,
                                        "rel": {},
                                        "event": []
                                    }
                                    new_line["text"] = ' '.join(text)
                                    new_line["bg"] = " "
                                    new_line["img_id"] = img_id
                                    new_line["img_id_dif"] = img_id_dif
                                    # print(new_line)

                                    # 将 new_line 写入 JSON 文件
                                    json_str = json.dumps(new_line)
                                    outfile.write(json_str + '\n')
                                    # json.dump(new_line, outfile, ensure_ascii=False, indent=4)
                                    text, labels = [], []

    def merged(self):
        twitter15 = self.path_out + "NER/data/twitter2015/"
        twitter17 = self.path_out + "NER/data/twitter2017/"
        mre = self.path_out + "RE/data/mre/"
        dirs = [twitter15, twitter17, mre]
        train = [dir + "train.jsonl" for dir in dirs]
        val = [dir + "dev.jsonl" for dir in dirs]
        test = [dir + "test.jsonl" for dir in dirs]
        dict = [train, val, test]

        output_dir = self.path_out + "merged/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        outfiles = [output_dir + "train.jsonl", output_dir + "dev.jsonl", output_dir + "test.jsonl"]

        for i, files in enumerate(dict):
            with open(outfiles[i], 'w') as outfile:
                print(f"write to {outfiles[i]}")

                for file_name in files:
                    with open(file_name, 'r') as infile:
                        for line in infile:
                            line = json.loads(line)  # 解析 JSON 行数据
                            json_str = json.dumps(line)
                            outfile.write(json_str + '\n')

    def replace(self, line):
        line = line.replace("'\"-@KellyCobiella'", "\"-@KellyCobiella\"")
        line = line.replace("'\"-'", "\"-\"")
        line = line.replace("'\"'", "\"'\"")
        line = line.replace("'O \" Santo'", "\"O ' Santo\"")
        line = line.replace("'is:\"D'", "\"is:'D\"")
        line = line.replace(" 'Miss \" Fingertoes'", " \"Miss ' Fingertoes\"")
        line = line.replace("'Bitch\"---'", "\"Bitch'---\"")
        line = line.replace("'America Bitch\"---'", "\"America Bitch'---\"")
        line = line.replace("'Adam \" Killa'", "\"Adam ' Killa\"")
        line = line.replace("'Nicholas \" Duffy \" Fudge'", "\"Nicholas  Duffy  Fudge\"")
        line = line.replace("\"Knockin ' On Heaven 's Door\"", "\"Knockin  On Heaven 's Door\"")
        line = line.replace("'Sol \" Campbell'", "\"Sol ' Campbell\"")

        line = line.replace("@", "AITE")
        line = line.replace(".", "dOT")
        line = line.replace("/", "XIEGANG")
        line = line.replace("\\", "FANXIEGANG")
        line = line.replace(":", "MAOHAO")
        line = line.replace(";", "FENHAO")
        line = line.replace(" ", "KONGGE")
        line = line.replace(",", "DOUHAO")
        line = line.replace("-", "HENGGANG")
        line = line.replace("!", "GANTANHAO")
        line = line.replace("?", "WENHAO")
        line = line.replace("*", "XINGHAO")
        line = line.replace("&", "ZAIYIQI")
        line = line.replace("(", "ZUOKUOHAO")
        line = line.replace("{", "ZUOJIHE")
        line = line.replace("[", "ZUOFANGHAO")
        line = line.replace(")", "YOUKUOHAO")
        line = line.replace("]", "YOUFANGHAO")
        line = line.replace("}", "YOUJIHE")
        line = line.replace("#", "JINGHAO")
        line = line.replace("%", "BAIFENHAO")
        line = line.replace("$", "MEIYUAN")
        line = line.replace("+", "JIAHAO")
        line = line.replace("~", "BOLANGHAO")
        line = line.replace("|", "SHUXIAN")
        line = line.replace("^", "ZHEHAO")
        line = line.replace("-", "JIANHAO")
        line = line.replace("=", "DENGYUHAO")
        line = line.replace("`", "XIEDIAN")
        line = re.sub(r"'(\w+)'", r'"\1"', line)
        line = line.replace("AITE", "@")
        line = line.replace("dOT", ".")
        line = line.replace("XIEGANG", "/")
        line = line.replace("FANXIEGANG", "\\")
        line = line.replace("MAOHAO", ":")
        line = line.replace("FENHAO", ";")
        line = line.replace("KONGGE", " ")
        line = line.replace("DOUHAO", ",")
        line = line.replace("HENGGANG", "-")
        line = line.replace("GANTANHAO", "!")
        line = line.replace("WENHAO", "?")
        line = line.replace("XINGHAO", "*")
        line = line.replace("ZAIYIQI", "&")
        line = line.replace("ZUOKUOHAO", "(")
        line = line.replace("ZUOFANGHAO", "[")
        line = line.replace("ZUOJIHE", "{")
        line = line.replace("YOUKUOHAO", ")")
        line = line.replace("YOUFANGHAO", "]")
        line = line.replace("YOUJIHE", "}")
        line = line.replace("JINGHAO", "#")
        line = line.replace("BAIFENHAO", "%")
        line = line.replace("MEIYUAN", "$")
        line = line.replace("JIAHAO", "+")
        line = line.replace("BOLANGHAO", "~")
        line = line.replace("SHUXIAN", "|")
        line = line.replace("ZHEHAO", "^")
        line = line.replace("JIANHAO", "-")
        line = line.replace("DENGYUHAO", "=")
        line = line.replace("XIEDIAN", "`")
        return line


if __name__ == "__main__":
    try:
        path_in = sys.argv[1]
        path_out = sys.argv[2]
        none = sys.argv[3]
        hyper_rel = sys.argv[4]
        inst = sys.argv[5]
    except:
        path_in = "./"
        path_out = "data/test/"
        none = "T"
        hyper_rel = "F"
        inst = "F"

    preprocessor = Pre_Process(path_in, path_out, [none, hyper_rel, inst])
    preprocessor.process()
