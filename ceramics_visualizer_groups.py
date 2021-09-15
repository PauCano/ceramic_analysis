import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import glob
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

colors = [mcolors.TABLEAU_COLORS["tab:blue"],
          mcolors.TABLEAU_COLORS["tab:orange"],
          mcolors.TABLEAU_COLORS["tab:green"],
          mcolors.TABLEAU_COLORS["tab:red"],
          mcolors.TABLEAU_COLORS["tab:purple"],
          mcolors.TABLEAU_COLORS["tab:brown"],
          mcolors.TABLEAU_COLORS["tab:pink"],
          mcolors.TABLEAU_COLORS["tab:gray"],
          mcolors.TABLEAU_COLORS["tab:olive"],
          mcolors.TABLEAU_COLORS["tab:cyan"]]
nColors = len(colors)
lineStyles = ["solid",
              "dotted",
              "dashed",
              "dashdot"]


spectrumFolderPath = "D:/Ceramica/pieces_dataV3/spectrum/"
classes = list(dict.fromkeys([group.split("\\")[-1].split("_")[0][1] for group in glob.glob(spectrumFolderPath+"g*.png")]))

xValues = [397.66, 400.28, 402.9, 405.52, 408.13, 410.75, 413.37, 416.0, 418.62, 421.24, 423.86, 426.49, 429.12, 431.74, 434.37, 437.0, 439.63, 442.26, 444.89, 447.52, 450.16, 452.79, 455.43, 458.06, 460.7, 463.34, 465.98, 468.62, 471.26, 473.9, 476.54, 479.18, 481.83, 484.47, 487.12, 489.77, 492.42, 495.07, 497.72, 500.37, 503.02, 505.67, 508.32, 510.98, 513.63, 516.29, 518.95, 521.61, 524.27, 526.93, 529.59, 532.25, 534.91, 537.57, 540.24, 542.91, 545.57, 548.24, 550.91, 553.58, 556.25, 558.92, 561.59, 564.26, 566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583.0, 585.68, 588.36, 591.04, 593.73, 596.41, 599.1, 601.78, 604.47, 607.16, 609.85, 612.53, 615.23, 617.92, 620.61, 623.3, 626.0, 628.69, 631.39, 634.08, 636.78, 639.48, 642.18, 644.88, 647.58, 650.29, 652.99, 655.69, 658.4, 661.1, 663.81, 666.52, 669.23, 671.94, 674.65, 677.36, 680.07, 682.79, 685.5, 688.22, 690.93, 693.65, 696.37, 699.09, 701.81, 704.53, 707.25, 709.97, 712.7, 715.42, 718.15, 720.87, 723.6, 726.33, 729.06, 731.79, 734.52, 737.25, 739.98, 742.72, 745.45, 748.19, 750.93, 753.66, 756.4, 759.14, 761.88, 764.62, 767.36, 770.11, 772.85, 775.6, 778.34, 781.09, 783.84, 786.58, 789.33, 792.08, 794.84, 797.59, 800.34, 803.1, 805.85, 808.61, 811.36, 814.12, 816.88, 819.64, 822.4, 825.16, 827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52, 847.29, 850.06, 852.83, 855.6, 858.37, 861.14, 863.92, 866.69, 869.47, 872.25, 875.03, 877.8, 880.58, 883.37, 886.15, 888.93, 891.71, 894.5, 897.28, 900.07, 902.86, 905.64, 908.43, 911.22, 914.02, 916.81, 919.6, 922.39, 925.19, 927.98, 930.78, 933.58, 936.38, 939.18, 941.98, 944.78, 947.58, 950.38, 953.19, 955.99, 958.8, 961.6, 964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27, 984.09, 986.9, 989.72, 992.54, 995.35, 998.17, 1000.99, 1003.81]
sides = ["A", "B", "C"]
classesToDisplay = []
sidesToDisplay = []#[False]*len(sides)
showSeparateSides = []
#classesToDisplay[0]=True
#classesToDisplay[4]=True
#sidesToDisplay[0]=True

def plotList(fileList, index, fig, ax):
    arrayWidth = io.imread(fileList[0]).shape[1]
    meanSet = np.zeros((len(fileList), arrayWidth))
    stdSet = np.zeros((len(fileList), arrayWidth))
    for i in range(len(fileList)):
        data = io.imread(fileList[i])
        meanSet[i] = data[0,:,0]/255*100
        stdSet[i] = data[0,:,1]/255*100
    mean = np.mean(meanSet, axis=0)
    std = np.mean(stdSet, axis=0)
    #ax.set_title(fileList[i].split("\\")[1].split(".")[0]+"_"+side, pad=20,fontdict={"weight":"bold"})
    color = colors[index%nColors]
    linestyle = lineStyles[index//nColors]
    file = fileList[0].split("\\")[1].split(".")[0]
    group = file.split("_")[0][1]
    side = file.split("_")[-1][1]
    label=group+"_"+side
    ax.plot(xValues, mean, color = color, linestyle = linestyle,label=label)
    ax.fill_between(xValues,mean-std,mean+std, alpha = 0.2, color = color, linestyle = linestyle)
    
    
class mainWindow(ttk.Frame):
    
    def __init__(self, root):
        super().__init__()
        self.initUI(root)
        graph = graphDisplayer(self)
    def initUI(self, root):
        global spectrumFolderPath
        self.master.title("Ceramic Spectrum Visualizer")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        pathFrame = ttk.Frame(self, height=50).grid(row=0, column=0, sticky = tk.NW)
        settingsFrame = ttk.Frame(self, height = 50).grid(row=1, column=0, rowspan=2, sticky = tk.NW)
        graphFrame = ttk.Frame(self).grid(row=0, column=2, sticky = tk.NW, rowspan=100)
        graph = graphDisplayer(graphFrame).grid(row=0, column=2, rowspan=100)
        
        
        spectrumLabel = tk.Label(pathFrame, text="Spectrum Folder Path").grid(row=0, column=0)
        spectrumPathEntry = tk.Entry(pathFrame, width=50)
        spectrumPathEntry.grid(row=1, column=0, columnspan=2)
        spectrumPathEntry.insert(0,spectrumFolderPath)
        spectrumFolderPath = spectrumPathEntry.get()
        #sidesFrame = ttk.Frame(self).grid(row=0, column=2)
        xPos=0
        displayLabelSides = tk.Label(settingsFrame, text = "Sides to Display").grid(row=2, column=0)
        for side in sides:
            pairFrame = ttk.Frame(settingsFrame)
            pairFrame.grid(row = xPos+3, column=1)
            label = tk.Label(pairFrame, text = side).pack(side="left")#.grid(row=yPos+2, column=0, sticky = tk.NE)
            box = tk.Checkbutton(pairFrame, variable = sidesToDisplay[xPos], width=10).pack(side="left")#.grid(row=yPos+2, column=1, sticky = tk.NW)
            xPos+=1
        pairFrame = ttk.Frame(settingsFrame)
        pairFrame.grid(row = xPos+3, column=1)
        label = tk.Label(pairFrame, text = "Graph each \nside separately").pack(side="left")#.grid(row=yPos+2, column=0, sticky = tk.NE)
        box = tk.Checkbutton(pairFrame, variable = showSeparateSides, width=10).pack(side="left")#.grid(row=yPos+2, column=1, sticky = tk.NW)
        
        yPos = 0
        displayLabel = tk.Label(settingsFrame, text = "Classes to Display").grid(row=2, column=0)
        for elClass in classes:
            pairFrame = ttk.Frame(settingsFrame)
            pairFrame.grid(row=yPos+3, column=0)
            label = tk.Label(pairFrame, text = elClass).pack(side="left")#.grid(row=yPos+2, column=0, sticky = tk.NE)
            box = tk.Checkbutton(pairFrame, variable = classesToDisplay[yPos], width=10).pack(side="left")#.grid(row=yPos+2, column=1, sticky = tk.NW)
            #pairFrame.pack("N")
            #pairFrame.pack(side="top")
            yPos+=1
            #label.grid()
    
class graphDisplayer(ttk.Frame):
    def __init__(self, parent):
        super().__init__()
        self.fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
        self.ax = axes
        button = tk.Button(self, text="Update Graph", command=self.updateGraph)
        button.pack()
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.updateGraph()        
        self.canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand=True)
    def updateGraph(self):
        self.ax.clear()
        for index in range(len(classes)):#elClass in classes:
            elClass = classes[index]
            if classesToDisplay[index].get():
                if showSeparateSides[0].get():
                    for sideIndex in range(len(sides)):
                        side = sides[sideIndex]
                        if sidesToDisplay[sideIndex].get():
                            fileList = glob.glob(spectrumFolderPath+"g"+elClass+"*"+side+".png")
                            plotList(fileList, index, self.fig, self.ax)
                    
                else:
                    fileList = []
                    for sideIndex in range(len(sides)):
                        side = sides[sideIndex]
                        if sidesToDisplay[sideIndex].get():
                            fileList.extend(glob.glob(spectrumFolderPath+"g"+elClass+"*"+side+".png"))
                    plotList(fileList, index, self.fig, self.ax)
        
        self.ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0))
        self.ax.set_xlabel("Wavelength(nm)")
        self.ax.set_ylabel("reflectance(%)")
        self.ax.set_ylim(0,100)
        self.fig.tight_layout()
        self.canvas.draw()
                    
        
        
                
                
                
def main():
    root = tk.Tk()
    root.geometry(str(1000)+"x"+str(700))
    for i in range(len(classes)):
        classesToDisplay.append(tk.BooleanVar(value=False))
    for i in range(len(sides)):
        sidesToDisplay.append(tk.BooleanVar(value=False))
    showSeparateSides.append(tk.BooleanVar(value=False))
    app = mainWindow(root)
    root.mainloop()
main()
                
                
                
                
                
                
                
            
            
            
            
            
            
            
            
            
            
'''           
                    fileList = glob.glob("./pieces_dataV3/spectrum/g"+elClass+"*"+side+".png")
                    
                    fig, axes = plt.subplots(nrows=math.ceil(len(fileList)/6), ncols=6, figsize=(80,40),sharey=False)
                    fig.suptitle(fileList[0].split("\\")[1].split(".")[0].split("_")[0], fontsize=32)
                    for i in range(len(fileList)):
                        data = io.imread(fileList[i])
                        mean = data[0,:,0]
                        std = data[0,:,1]
                        #print(i, len(fileList), math.ceil(len(fileList)/6))
                        ax = axes.flatten()[i]
                        ax.set_title(fileList[i].split("\\")[1].split(".")[0]+"_"+side, pad=20,fontdict={"weight":"bold"})
                        ax.plot(xValues, mean)
                        ax.fill_between(xValues,mean-std,mean+std, alpha = 0.2)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
                    plt.show()
                    
                    
                    
            fileList = glob.glob("./pieces_dataV3/spectrum/g"+elClass+"*A.png")
            #print(fileList)
            
            fig, axes = plt.subplots(nrows=math.ceil(len(fileList)/6), ncols=6, figsize=(80,40),sharey=False)
            fig.suptitle(fileList[0].split("\\")[1].split(".")[0].split("_")[0], fontsize=32)
            for i in range(len(fileList)):
                data = io.imread(fileList[i])
                mean = data[0,:,0]
                std = data[0,:,1]
                #print(i, len(fileList), math.ceil(len(fileList)/6))
                ax = axes.flatten()[i]
                ax.set_title(fileList[i].split("\\")[1].split(".")[0], pad=20,fontdict={"weight":"bold"})
                ax.plot(xValues, mean)
                ax.fill_between(xValues,mean-std,mean+std, alpha = 0.2)
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.show()
'''