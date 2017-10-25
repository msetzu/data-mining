import numpy as np
import matplotlib.pyplot as pp

import settings
from settings import data, categorical_labels_pretty_prints, departments_pretty_prints, departments


def draw_discrete_distribution_tuples(vars, var_pretty_prints, title, ticks, colors=settings.palette):
    """
	Draw the distcrete variables' distributions.
    :param vars:                Variables to draw. distribution to draw. Dictionary: label -> value
	:param var_pretty_print: 	The bars' pretty print names. The x axis ticks.
	:param title:				The graph's title.
	:param ticks:				The x axis ticks.
	:return: 					Nothing.
	"""
    n = len(ticks)
    width = 0.8*(0.9/len(var_pretty_prints))
    index = np.arange(n)

    figure, axes = pp.subplots()
    offsets = list(map(lambda x: index + width * x[1], list(zip([width] * n, list(range(n))))))
    for var, pretty_print, color_key, offset in zip(vars, var_pretty_prints, colors, offsets):
        axes.bar(x=offset, height=vars[var], color=colors[color_key], width=width, label=pretty_print)

    axes.set_xticks(index + width / len(var_pretty_prints))
    axes.set_xticklabels(ticks, rotation=25, ha='right')
    axes.legend()

    pp.tight_layout()
    pp.title(title)
    pp.savefig(str(title) + '.png')


def salary_per_department(hr, left=False):
    """
    Draw graph with salary distribution per department. If left = True, also add the number of people who left.
    :param hr:      The data object.
    :param left:    True for drawing left rate, false otherwise. Default: False
    :return:        Nothing.
    """
    departments = list(set(hr.data["sales"]))
    departments.sort()
    sales_int = {k: v for k, v in zip(departments, range(len(departments)))}
    sales_int = list(map(lambda x: sales_int[x], list(data["sales"])))
    settings.data = data.assign(sales_int = sales_int)

    department_salary_distribution = {}
    chart_values = []
    buckets = ["low", "medium", "high"]

    for bucket in buckets:
        department_salary_distribution[bucket] = []
        for department in departments:
            employees = len(data[(data["sales"] == department) & (data["salary"] == bucket)])
            department_salary_distribution[bucket].append(employees)

    department_salary_distribution["left"] = []
    department_salary_distribution["left"] = [len(data[(data["sales"] == department) & (data["left"] == 1)]) for department in departments]

    if left:
        draw_discrete_distribution_tuples(vars=department_salary_distribution, var_pretty_prints=buckets + ["left"], title="Salary per department, with left rate", ticks=departments_pretty_prints, colors=settings.large_palette)
    else:
        draw_discrete_distribution_tuples(vars=chart_values, var_pretty_prints=buckets, title="Salary per department", ticks=departments_pretty_prints, colors=settings.large_palette)


def promotions_per_project(hr):
    projects_range = range(min(list(hr.data["number_project"])), max(list(hr.data["number_project"])) + 1)
    promotions_per_project = {}

    for bucket in [1, 0]:
        promotions_per_project[bucket] = []
        for nr_project in projects_range:
            promotions_per_project[bucket].append(data[(data["promotion_last_5years"] == bucket) & (data["number_project"] == nr_project)]["number_project"].shape[0])

    draw_discrete_distribution_tuples(vars=promotions_per_project,
                                      var_pretty_prints=list(categorical_labels_pretty_prints["promotion_last_5years"]),
                                      title="Promotion according to number of projects",
                                      ticks=list(map(str, projects_range)),
                                      colors=settings.palette)

    pp.title("Promotion rate per number of projects")
    pp.savefig("Promotion rate per number of projects.png")
    pp.draw()