import os
import tkinter
import lightgbm as lgb
from tkinter import filedialog
from tkinter import ttk
from features import FeatureExtractor

MODEL_PATH = os.path.join(os.getcwd(), "model.txt")
THRESHOLD = 0.5


class CinderGUI(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.parent = parent
        self.threshold = THRESHOLD
        self.model = lgb.Booster(model_file=MODEL_PATH)
        self.extractor = FeatureExtractor()

        parent.title("Cinder")
        parent.geometry("800x600")
        parent.resizable(width=False, height=False)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        parent.option_add('*tearOff', 'FALSE')
        self.grid(column=0, row=0, sticky='nsew')

        self.grid_columnconfigure(0, weight=1)
        label = ttk.Label(
            self,
            text="Cinder - A tiny Machine learning-based Malware Detector",
            font='Arial 24 bold')
        label.grid(row=0, column=0, columnspan=2, sticky='nsew')
        label.configure(anchor="center")
        btn_scan = ttk.Button(self, text='Scan', command=self.scan)
        btn_scan.grid(row=1, column=0, sticky='ew')
        btn_reset = ttk.Button(self, text='Clear', command=self.clear)
        btn_reset.grid(row=1, column=1, sticky='ew')
        self.table_result = ttk.Frame(self)
        self.table_result.grid(row=2, column=0, columnspan=2)

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

    def scan(self):
        selected_files = filedialog.askopenfilenames(
            parent=self,
            title="Choose PE files",
            filetypes=(("PE Files",
                        (".acm", ".ax", ".cpl", ".dll",
                         ".drv", ".efi", ".exe", ".mui",
                         ".ocx", ".scr", ".sys", ".tsp")),))

        if len(selected_files) == 0:
            return

        cell_index = ttk.Label(self.table_result, text="#", font='Arial 16 bold')
        cell_index.grid(row=0, column=0, sticky='nsew')
        cell_index.configure(anchor="center")
        cell_name = ttk.Label(self.table_result, text="Filename", font='Arial 16 bold')
        cell_name.grid(row=0, column=1, sticky='nsew')
        cell_name.configure(anchor="center")
        cell_score = ttk.Label(self.table_result, text="Predict", font='Arial 16 bold')
        cell_score.grid(row=0, column=2, sticky='nsew')
        cell_score.configure(anchor="center")

        feature_vectors = [self.extractor.feature_vector(file_path) for file_path in selected_files]
        predict_values = self.model.predict(feature_vectors)

        for idx, file_path in enumerate(selected_files):
            file_name = os.path.basename(file_path)
            file_score = predict_values[idx]

            cell_index = ttk.Label(self.table_result, text=str(idx))
            cell_index.grid(row=idx + 1, column=0, sticky='nsew', padx=10)
            cell_index.configure(anchor="w")
            cell_name = ttk.Label(self.table_result, text=file_name)
            cell_name.grid(row=idx + 1, column=1, sticky='nsew', padx=10)
            cell_name.configure(anchor="w")
            cell_score = ttk.Label(self.table_result, text='{:2.4f} %'.format(file_score * 100))
            if file_score >= self.threshold:
                cell_score.configure(foreground="red")
            cell_score.configure(anchor="e")
            cell_score.grid(row=idx + 1, column=2, sticky='nsew', padx=10)

    def clear(self):
        for child in self.table_result.winfo_children():
            child.destroy()


if __name__ == '__main__':
    root = tkinter.Tk()
    CinderGUI(root)
    root.mainloop()
