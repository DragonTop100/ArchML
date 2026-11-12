import matplotlib.pyplot as plt


def _bar_plot(results, key, ylabel, title, value_fmt='{:.1f}',
              ylim=None, save_path=None):
    methods = list(results.keys())
    values = [results[m][key] for m in methods]

    fig, ax = plt.subplots()
    ax.bar(methods, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)

    for i, v in enumerate(values):
        ax.text(i, v, value_fmt.format(v), ha='center', va='bottom')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_size_vs_method(results, save_path=None):
    return _bar_plot(
        results, key='size_mb',
        ylabel='Размер модели, МБ',
        title='Размер модели по методу сжатия',
        save_path=save_path,
    )


def plot_latency_vs_method(results, save_path=None):
    return _bar_plot(
        results, key='latency_ms',
        ylabel='Время обработки 1 изображения, мс',
        title='Скорость работы по методу сжатия',
        save_path=save_path,
    )


def plot_accuracy_vs_method(results, save_path=None):
    scaled = {m: {'accuracy_pct': v['accuracy'] * 100}
              for m, v in results.items()}
    return _bar_plot(
        scaled, key='accuracy_pct',
        ylabel='Точность, %',
        title='Точность по методу сжатия',
        ylim=(0, 100),
        save_path=save_path,
    )


def plot_learning_curve(fractions, accuracies, save_path=None):
    fig, ax = plt.subplots()
    x = [f * 100 for f in fractions]
    y = [a * 100 for a in accuracies]

    ax.plot(x, y, marker='o')
    ax.set_xlabel('Доля обучающих данных, %')
    ax.set_ylabel('Точность на val, %')
    ax.set_title('Кривая обучения')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig
