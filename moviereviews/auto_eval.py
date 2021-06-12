import subprocess

base_cmd = [ "python", "sentiment_analysis.py" ]

f = open('auto_eval_3_12063.txt', 'w')

for i in range(0, 100):
    r = (i+1) / 100
    print('param:', r, '\t', end='', flush=True)

    f.write(str(r))
    f.write('\t')

    subprocess.run(base_cmd + [ "-s", "3", "-m", "train", "-i", "datasets/train.tsv", "-n", "model/temp_auto.model", "-f", str(r) ])
    subprocess.run(base_cmd + [ "-s", "3", "-m", "predict", "-i", "datasets/dev.tsv", "-n", "model/temp_auto.model", "-r", "predictions/temp_auto.tsv" ])
    res = subprocess.run(base_cmd + [ "-s", "3", "-m", "eval", "-i", "datasets/dev.tsv", "-r", "predictions/temp_auto.tsv" ], stdout=subprocess.PIPE)
    resstr = res.stdout.decode("utf-8").strip()
    print(resstr)
    f.write(resstr.replace('accuracy: ', ''))
    f.write('\n')
    f.flush()

f.close()
