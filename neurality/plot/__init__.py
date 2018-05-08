region_color_mapping = {'V4': '#00cc66', 'IT': '#ff9000'}


def shaded_errorbar(x, y, error, ax, alpha=0.4, **kwargs):
    ax.plot(x, y)
    ax.fill_between(x, y - error, y + error, alpha=alpha, **kwargs)
