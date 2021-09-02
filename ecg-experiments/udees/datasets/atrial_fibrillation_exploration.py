from udees.datasets.mitdb import AtrialFibrillation
from matplotlib import pyplot as plt

#ALI: AtrialFibrillation.download()
records = AtrialFibrillation.get_records()
# print(len(records)) # 16

samples, labels = records[1].as_uniformly_sized_examples(
                  downsampling_factor=2,
                  example_window_size=250*10,
                  offset_factor=0
       )
# print(len(samples)) #3675
# print(len(labels)) #3675


nrows = 2
ncols = 2
fig, axis = plt.subplots(nrows, ncols, sharex=True, sharey=True, dpi=300)
factor = 415



for i in range(nrows):
    for j in range(ncols):
        k = factor*(2*j+2*i+1)
        annotation = labels[k]
        if j % 2 == 0:
            while annotation != "(AFIB":
                k = k + 1
                annotation = labels[k]
        else:
            while annotation != "(N":
                k = k + 1
                annotation = labels[k]


        print(k)
        axis[i][j].plot(samples[k][:400, 0]) #slicing here to tell which part of the graph to show
        title = "Atrial fibrillation" if annotation == "(AFIB" else "Sinus rhythm"
        if i == 1:
            axis[i][j].set_title(title)

# print(samples[0].shape) #(1250,2)



fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
plt.xlabel("Time [8ms]")
plt.ylabel("ECG amplitude [c * V]",labelpad = 15)

plt.savefig("atrial_fib_vs_sinus.pdf")
