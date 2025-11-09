import os
import subprocess
import sacrebleu

LANG_PAIR = "fr-en"  

DATA_DIR = f"data/{LANG_PAIR}/dataset"
MODELS_DIR = f"data/{LANG_PAIR}/models"
SRC_FILE = os.path.join(DATA_DIR, "test.src") 
REF_FILE = os.path.join(DATA_DIR, "test.tgt")  
GPU = "0"

with open(REF_FILE, "r", encoding="utf-8") as f:
    refs = [line.strip() for line in f.readlines()]

results = []

for filename in sorted(os.listdir(MODELS_DIR)):
    if filename.endswith(".pt"):
        step = filename.replace("model_step_", "").replace(".pt", "")
        model_path = os.path.join(MODELS_DIR, filename)
        output_path = f"data/{LANG_PAIR}/test_{step}.hyp"

        print(f"Translating with model {filename}...")

        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{os.getcwd()}/data:/opt/opennmt-py/data",
            "-v", f"{os.getcwd()}/config/{LANG_PAIR}.yaml:/opt/opennmt-py/config.yaml",
            "opennmt-py",
            "onmt_translate",
            "-model", f"data/{LANG_PAIR}/models/{filename}",
            "-src", SRC_FILE,
            "-output", output_path,
            "-gpu", GPU,
            "-replace_unk"
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open(output_path, "r", encoding="utf-8") as f:
            hyps = [line.strip() for line in f.readlines()]

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        ter = sacrebleu.corpus_ter(hyps, [refs]).score

        print(f"Step {step}: BLEU: {bleu:.2f}, TER: {ter:.2f}")
        results.append((int(step), bleu, ter, output_path))

best_model = max(results, key=lambda x: x[1])
print(f"\n Best model for {LANG_PAIR}: model_step_{best_model[0]}.pt")
print(f"BLEU: {best_model[1]:.2f} | TER: {best_model[2]:.2f}")
