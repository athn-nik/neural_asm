import pickle

with open('words_in_men.pkl', 'rb') as fl:
    nns = pickle.load(fl)

for split in ["train", "dev", "test"]:
    # targ_out = open(os.path.join(args.out_folder, "targ-"+split+".txt"), "w")
    # label_out = open(os.path.join(args.out_folder, "label-"+split+".txt"), "w")
    print('*******************' + split + '*******************\n')
    ll = 0

    with open("src-" + split + ".txt", "r") as fl:
        for ind, line in enumerate(fl):
            if ind != 0:
                d = line.strip().split("\t")

                label = d[2]
                c1 = 0
                for wd in d[0].strip().split():
                    if wd in nns:
                        c1 += 1

                c2 = 0
                for wd in d[1].strip().split():
                    if wd in nns:
                        c2 += 1
                if c2 >= 3 and c1 >= 3:
                    ll += 1
                    f = open('three_common_at_least' + split + '.txt', 'a')
                    f.write(line)
                    f.close()
    print(ll)
