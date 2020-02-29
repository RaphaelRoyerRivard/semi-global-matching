import os
import numpy as np
from matplotlib import pyplot as plt


def autolabel(rects, ax, precision):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(('{:.' + str(precision) + 'f}').format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=7)


def parse_results():
    results = {}
    for path, subfolders, files in os.walk("."):
        if "result.txt" not in files:
            continue

        f = open(f'{path}/result.txt', 'r')
        lines = f.readlines()
        f.close()

        split_folder = path.split("\\")[-1].split("_")
        img = split_folder[0]
        descriptor = split_folder[-1]

        left = lines[0].split(";")
        left = (float(left[0]), float(left[1]))
        right = lines[1].split(";")
        right = (float(right[0]), float(right[1]))

        if img not in results.keys():
            results[img] = {}
        results[img][descriptor] = (left, right)

    print(results)
    dataset_groups = {
        # "2003": (["cones", "teddy"], []),
        # "2005": (["Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer"], [])
        "baseline": (["Art", "Books", "Dolls"], []),
        "intensity": (["ArtI", "BooksI", "DollsI"], []),
        "texture": (["Cloth1", "Wood1", "Wood2"], [])
    }
    recall = np.zeros((len(results), 2))
    bmpre = np.zeros((len(results), 2))
    for i, img in enumerate(results.keys()):
        for dataset_group in dataset_groups.keys():
            if img in dataset_groups[dataset_group][0]:
                dataset_groups[dataset_group][1].append(i)
        for j, descriptor in enumerate(results[img].keys()):
            res = results[img][descriptor]
            recall[i, j] = (res[0][0] + res[1][0]) / 2
            bmpre[i, j] = (res[0][1] + res[1][1]) / 2

    labels = list(results.keys())

    bar_width = 0.25

    for metric in ['Rappel', 'BMPRE (Bad Matching Pixels Relative Error)']:
        values = recall if metric == 'Rappel' else bmpre
        score_label = 'Scores' + (' (le plus bas est le mieux)' if metric.startswith("BMPRE") else '')
        precision = 2 if metric == 'Rappel' else 3

        # Graph 1: separated for image pair
        x = np.arange(len(labels))
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - bar_width / 2, values[:, 0], bar_width, label='BRIEF')
        rects2 = ax.bar(x + bar_width / 2, values[:, 1], bar_width, label='HOG')
        ax.set_xlabel("Paires d\'images")
        ax.set_ylabel(score_label)
        ax.set_title(metric + " pour chaque paire d'images")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        autolabel(rects1, ax, precision)
        autolabel(rects2, ax, precision)
        fig.tight_layout()
        plt.show()

        # Graph 2: averaged by dataset group
        x = np.arange(len(dataset_groups))
        fig, ax = plt.subplots()
        brief_values_by_group = [values[dataset_groups[group][1], 0].mean() for group in dataset_groups]
        hog_values_by_group = [values[dataset_groups[group][1], 1].mean() for group in dataset_groups]
        rects1 = ax.bar(x - bar_width / 2, brief_values_by_group, bar_width, label='BRIEF')
        rects2 = ax.bar(x + bar_width / 2, hog_values_by_group, bar_width, label='HOG')
        ax.set_xlabel('Groupes de base de données')
        ax.set_ylabel(score_label)
        ax.set_title(metric + " moyen par groupe de base de données")
        ax.set_xticks(x)
        ax.set_xticklabels(list(dataset_groups.keys()))
        ax.legend()
        autolabel(rects1, ax, precision)
        autolabel(rects2, ax, precision)
        fig.tight_layout()
        plt.show()

        # Graph 3: averaged on all image pairs
        bar_values = [values[:, desc].mean() for desc in range(2)]
        bar_labels = ['BRIEF', 'HOG']
        x = np.arange(len(bar_values))
        rects = plt.bar(x, bar_values, align='center')
        ax = plt.gca()
        autolabel(rects, ax, precision)
        plt.xticks(x, bar_labels)
        plt.title(metric + " moyen sur toutes les paires d'images")
        plt.xlabel('Méthodes')
        plt.ylabel(score_label)
        print(bar_values)
        plt.show()


if __name__ == '__main__':
    parse_results()