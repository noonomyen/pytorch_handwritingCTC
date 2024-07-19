from csv import writer

data = open("train.log", "r", encoding="utf-8").readlines()
outf = open("train.csv", "w", encoding="utf-8", newline="")
outw = writer(outf)
outw.writerow(("fit", "epoch", "train_loss", "valid_loss", "cer", "ier", "train_leven", "val_leven"))

fit = 0
old_epoch = 2
sum_epoch = 0

for idx, line in enumerate(data):
    if (_ := line.strip()) != "\n" and _ != "":
        split = line.split(" ")
        epoch = int(split[1][:-1])

        if epoch < old_epoch:
            fit += 1

        old_epoch = epoch
        sum_epoch += 1

        outw.writerow((
            fit,
            sum_epoch,
            float(split[4]),
            float(split[8]),
            float(split[11]),
            float(split[14]),
            float(split[17]),
            float(split[21])
        ))
