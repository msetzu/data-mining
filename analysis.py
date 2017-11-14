from optparse import OptionParser
from matplotlib import mlab, patches as mpatches
import numpy as np
import matplotlib.pyplot as pp
from scipy.interpolate import spline
import itertools

import settings
from settings import *


def data_analysis(hr):
    parsed_arguments = parse_arguments()
    distributions = parsed_arguments["distributions"]

    if distributions.count("all") > 0:
        draw_distributions(hr, continuous_labels, pretty_prints[:2])
        draw_discrete_distributions(hr, discrete_labels, pretty_prints[2:5])
        draw_categorical_distributions(hr, categorical_labels, categorical_labels_pretty_prints)
    else:
        discrete_vars = set(distributions).intersection(discrete_labels)
        continuous_vars = set(distributions).intersection(continuous_labels)
        categorical_vars = set(distributions).intersection(categorical_labels)
        for var in discrete_vars:
            var_pretty_print = labels_pretty_print[var]
            draw_discrete_distribution(hr, var, var_pretty_print, var_pretty_print)
        for var in continuous_vars:
            var_pretty_print = labels_pretty_print[var]
            draw_distribution(hr, var, var_pretty_print)
        for var in categorical_vars:
            var_pretty_prints = categorical_labels_pretty_prints[var]
            title = labels_pretty_print[var]
            draw_categorical_distribution(hr, var, var_pretty_prints, title)

    if parsed_arguments["draw_correlation"]:
        draw_correlation_matrix(hr.normal[correlated_labels].corr(), pretty_prints[:5] + [pretty_prints[-1]],
                                normalised=True)
    if parsed_arguments["promotions-per-project"]:
        promotions_per_project(hr)
    if parsed_arguments["left-per-department"]:
        left_per_department(hr)
    if parsed_arguments["salary-per-department"]:
        salary_per_department(hr)
    if parsed_arguments["scatter-plots"]:
        draw_scatter_plots(hr)
    if parsed_arguments["log"]:
        var = "time_spend_company"
        line_plot_before(hr.data[var], labels_pretty_print[var])
        to_log = draw_discrete_log_transformation(hr, var, labels_pretty_print[var])   
        line_plot_after(to_log, labels_pretty_print[var]) 
        var="last_evaluation"
        line_plot_before(hr.data[var], labels_pretty_print[var])
        to_log = draw_discrete_log_transformation(hr, var, labels_pretty_print[var])
        line_plot_after(to_log, labels_pretty_print[var])
    if parsed_arguments["time-company-project"]:
        time_company_per_project(hr)    


def parse_arguments():
    arguments = {}
    parser = OptionParser()
    parser.add_option("-d", "--distribution", dest="distributions", help="List distributions to plot, comma separated", metavar="DISTRIBUTIONS")
    parser.add_option("--correlation", action="store_true", dest="correlation", help="Show correlation matrix")
    parser.add_option("--left-per-department", action="store_true", dest="left_per_department", help="Show expected and actual left rate")
    parser.add_option("--salary-per-department", action="store_true", dest="salary_per_department", help="Show salary per department, account for left employees")
    parser.add_option("--promotions-per-project", action="store_true", dest="promotions_per_project", help="Show promotions rate per number of project")
    parser.add_option("--scatter-plots", action="store_true", dest="scatter_plots", help="Plot scatter plots")
    parser.add_option("--log", action="store_true", dest="log", help="Plot log transformation")
    parser.add_option("--time-company-project", action="store_true", dest="time_company_per_project", help="Plota bar chart with numbers of project and time spend company")
   
    (options, args) = parser.parse_args()
    arguments["draw_correlation"] = options.correlation
    arguments["scatter-plots"] = options.scatter_plots
    arguments["left-per-department"] = options.left_per_department
    arguments["salary-per-department"] = options.salary_per_department
    arguments["promotions-per-project"] = options.promotions_per_project
    arguments["log"] = options.log
    arguments["time-company-project"] = options.time_company_per_project

    if not (options.distributions is None):
        arguments["distributions"] = options.distributions.split(",")
    else:
        arguments["distributions"] = list()

    return arguments


def draw_scatter_plots(hr):
    columns = correlated_labels
    tuples = list(itertools.combinations(columns, 2))
    for i, (var_x, var_y) in enumerate(tuples):
        print(str(i))
        df = (hr.discrete.groupby(var_x).apply(lambda x: x.sample(n=scatter["sampling_size"], replace=True)))[
            [var_x, var_y]]
        draw_scatter_plot(list(df[var_x]), list(df[var_y]),
                          labels_pretty_print[var_x], labels_pretty_print[var_y],
                          "Scatter-" + str(i))


def quantiles(data, quantiles):
    """
    Get the given vars quantiles for the given data.
    :param data 		The dataframe.
    :param quantiles 	List of quantiles to compute.
    :return				List of quantiles for the provided data.
    """
    return [data.quantile(q=quantile) for quantile in quantiles]


def draw_distributions(hr, vars, vars_pretty_print):
    for var, var_pretty_print in zip(vars, vars_pretty_print):
        draw_distribution(hr, var, var_pretty_print)


def draw_discrete_distributions(hr, vars, vars_pretty_print):
    for var, var_pretty_print, in zip(vars, vars_pretty_print):
        draw_discrete_distribution(hr, var, var_pretty_print, labels_pretty_print[var])


def draw_categorical_distributions(hr, vars, vars_pretty_prints):
    for var, var_pretty_print in zip(vars, vars_pretty_prints):
        draw_categorical_distribution(hr, var, categorical_labels_pretty_prints[var_pretty_print],
                                      labels_pretty_print[var])


def draw_distribution(hr, var, var_pretty_print):
    """
    Draw the variables' distributions.
    :param hr: 					The data object.
    :param var: 				The variable whose distribution to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :return: 					Nothing.
    """
    var_val = list(hr.data[var])
    var_val_std = hr.std[var]
    var_val_mean = hr.mean[var]
    num_bins = int(np.ceil(np.log2(len(var_val))) + 1)

    figure, axes = pp.subplots()

    n, bins, patches = axes.hist(var_val, num_bins, label='Data distribution', color=palette['main'], normed=False,
                                 stacked=True)
    y = mlab.normpdf(bins, var_val_mean, var_val_std) * sum(n * np.diff(bins))
    qs = quantiles(hr.data[var], list(np.arange(.25, 1, .25)))

    # Draw quantiles
    for index, quantile in enumerate(qs):
        axes.axvline(quantile, color=palette["pr_complementary"], linestyle="dashed")

    axes.plot(bins, y, '.-.', color=palette['secondary'], label="Gaussian approximation")
    axes.set_xlabel(str(var_pretty_print))
    axes.set_ylabel('Employees')
    axes.set_title(r'Distribution: $\mu$ =' + format(var_val_std, '.4f') + ', $\sigma$ =' + format(var_val_mean, '.4f'))

    handles, labels = axes.get_legend_handles_labels()
    quantiles_patch = mpatches.Patch(color=palette["pr_complementary"], label='Quartiles')
    handles.append(quantiles_patch)
    lgd = axes.legend(handles + [quantiles_patch], labels + ["Quartiles"], loc='lower center',
                      bbox_to_anchor=(.5, -.25), ncol=3)
    pp.title(str(var_pretty_print) + ': data distribution')
    pp.savefig(str(var_pretty_print) + '.png', bbox_extra_artists=[lgd], bbox_inches='tight')


def draw_categorical_distribution(var_val, var_pretty_prints, title):
    """
    Draw the distcrete variables' distributions.
    :param hr: 					The data object.
    :param var: 				The variable whose distribution to draw.
    :param var_pretty_prints: 	The variable's pretty print name.
    :param title:				The plot's title.
    :return: 					Nothing.
    """
    figure, axes = pp.subplots()
    x_values = list(set(var_val))
    x_ticks = range(len(x_values))
    x_values.sort()
    x_values.reverse()

    frequency = list()
    for x in x_values:
        frequency.append(var_val.count(x))

    if x_values == [1, 0]:
        x_ticks = list(var_pretty_prints)
        frequency.reverse()

    bars = pp.bar(x_ticks, frequency)
    bars[0].set_color(palette["main"])
    bars[1].set_color(palette["secondary"])
    axes.set_ylabel("Employees")
    axes.set_title(str(title))
    axes.set_xticklabels(x_ticks)
    pp.title(title)

    pp.tight_layout()
    pp.savefig(str(title) + '.png')


def draw_discrete_distribution(var_val, var_pretty_print, title):
    """
    Draw the distcrete variables' distributions.
    :param var_val: 			The values to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :param title: 				The plot's title.
    :return: 					Nothing.
    """
    figure, axes = pp.subplots()
    x_values = list(set(var_val))
    x_values.sort()

    frequency = list()
    for x in x_values:
        frequency.append(var_val.count(x))

    pp.bar(x_values, frequency, color=palette['main'])
    axes.set_xticks(x_values)
    axes.set_ylabel('Employees')
    axes.set_title(str(title))

    pp.tight_layout()
    pp.savefig(str(var_pretty_print) + '.png')


def draw_correlation_matrix(correlation_matrix, pretty_labels, normalised=False):
    figure, axes = pp.subplots()

    mask = np.tri(correlation_matrix.shape[0], k=-1)
    matrix = np.ma.array(correlation_matrix, mask=mask)  # mask out the lower triangle
    cmap_pale_pink.set_bad('w')
    image = axes.imshow(matrix, cmap=cmap_pale_pink, interpolation='none')
    pp.colorbar(image)

    axes.set_xticks(np.arange(0, correlation_matrix.shape[0], correlation_matrix.shape[0] * 1.0 / len(pretty_labels)))
    axes.set_yticks(np.arange(0, correlation_matrix.shape[1], correlation_matrix.shape[1] * 1.0 / len(pretty_labels)))
    axes.set_xticklabels(pretty_labels)
    axes.set_yticklabels(pretty_labels)
    axes.set_xticklabels(pretty_labels, rotation=45, ha='right')

    # Hide borders
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    pp.tight_layout()
    pp.draw()

    if not normalised:
        axes.set_title('Correlation matrix', fontsize=12, fontweight='bold')
        pp.savefig('Correlation matrix.png')
    else:
        axes.set_title('Correlation matrix', fontsize=12, fontweight='bold')
        pp.savefig('Correlation matrix for normalised data.png')


def draw_scatter_plot(var_x, var_y, var_x_pretty_print, var_y_pretty_print, title):
    """
    Draw the scatter plot of the two variables.
    :param var_x: 				The x axis values.
    :param var_y: 				The y axis values.
    :param var_x_pretty_print: 	The x axis pretty print.
    :param var_y_pretty_print: 	The y axis values.
    :param title: 				The plot's title.
    :return: 					Nothing.
    :Note: For the discrete features there is a hr.discretize in objects. For now only last_eval (but who knows)
    """
    figure, axes = pp.subplots()
    axes.scatter(var_x, var_y, color=palette["main"])
    axes.set_xlabel(var_x_pretty_print)
    axes.set_ylabel(var_y_pretty_print)
    pp.savefig(title + ".png")


def draw_discrete_distribution_stacked(vars, var_pretty_prints, title, ticks, legend=list(), colors=settings.large_palette_stacked, stack_pretty_print = "", top_text=list()):
    """
    Draw the distcrete variables' distributions with a stacked value.
    :param vars:                Variables to draw. Dictionary: label -> value.
                                Note that each value on the x axis has all the provided labels.
    :param var_pretty_prints: 	The bars' pretty print names. The x axis ticks.
    :param title:				The graph's title.
    :param ticks:				The x axis ticks.
    :param colors: 				The colormap for the graph.
    :param top_text:            Text to put on top of the bar. Default: none.
    :return: 					Nothing.
    """
    n = len(ticks)
    if len(legend) == 0:
        width = 0.8 * (0.9 / len(var_pretty_prints))
    else:
        width = 0.8 * (0.9 / len(legend))

    index = np.arange(n)

    figure, axes = pp.subplots()
    offsets = list(map(lambda x: index + width * x[1], list(zip([width] * n, list(range(n))))))

    if list(vars.keys()).count("stack") == 0:
        for var, pretty_print, color_key, offset in zip(vars, var_pretty_prints, colors, offsets):
            axes.bar(x=offset, height=vars[var], color=colors[color_key], width=width, label=pretty_print)

        if len(top_text) > 0:
            heights = list(map(lambda patch: patch.get_height(), axes.patches))
            for patch, height, text in zip(axes.patches, heights, top_text):
                axes.text(patch.get_x() + patch.get_width() / 2, height + 5, text, ha='center', va='bottom')

    else:
        keys = list(vars.keys())
        keys.remove("stack")

        for var, stack_var, pretty_print, color_key, offset in zip(keys, vars["stack"], var_pretty_prints, colors, offsets):
            axes.bar(x=offset, height=vars[var], color=colors[color_key], width=width, label=pretty_print.capitalize())
            axes.bar(x=offset, height=vars["stack"][var], color=large_palette_stacked["stack"], width=width, label="stack")

        handles, labels = axes.get_legend_handles_labels()
        labels_indexes = set(list(range(len(labels)))) - set([label[0] for label in filter(lambda x: x[0] % 2 == 1, enumerate(labels))])
        new_labels = [labels[i] for i in labels_indexes]
        new_handles = [handles[i] for i in labels_indexes]
        stack_patch = mpatches.Patch(color=large_palette_stacked["stack"], label=stack_pretty_print)
        new_handles.append(stack_patch)
        new_labels.append(stack_pretty_print)

    axes.set_xticklabels(ticks, rotation=25, ha="right")
    axes.set_xticks(index + width / len(var_pretty_prints))
    new_handles, new_labels = axes.get_legend_handles_labels()

    if len(legend) > 0:
        lgd = axes.legend(new_handles, legend, loc="best")
    else:
        lgd = axes.legend(new_handles, new_labels, loc="best")

    pp.title(str(title))
    pp.savefig(str(title) + '.png', bbox_extra_artists=[lgd], bbox_inches='tight')
    return


def left_per_department(hr):
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
    columns = ["salary_int", "sales", "left"]

    left_employees = hr.discrete[hr.discrete["left"] == 1][columns]
    left_employees_num = left_employees.shape[0]
    department_sizes = [(hr.data[hr.data["sales"] == department]).shape[0] for department in departments]
    department_sizes_percentages = list(map(lambda x: round(x/14999,2), department_sizes))
    left_employees_per_department = {department: left_employees[left_employees["sales"] == department][columns]
                                    for department in departments}
    left_employees_department = list(map(lambda x: x.shape[0], left_employees_per_department.values()))
    expected_left_employees = list(map(lambda percentage: int(percentage * left_employees_num), department_sizes_percentages))

    draw_discrete_distribution_stacked(vars = {"Actual leave": left_employees_department,
                                               "Expected leave": expected_left_employees},
                                                var_pretty_prints = departments_pretty_prints,
                                                title = "Left employees per department, expected left employees per department",
                                                ticks = departments_pretty_prints,
                                                legend = ["Left employees", "Expected left employees"],
                                                colors = palette)


def salary_per_department(hr):
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
    settings.data = data.assign(sales_int=sales_int)
    columns = ["salary_int", "sales", "left"]
    department_salary_distribution = {}
    department_salary_distribution["stack"] = {}
    buckets = ["low", "medium", "high"]

    for bucket in buckets:
        department_salary_distribution[bucket] = []
        department_salary_distribution["stack"][bucket] = []
        for department in departments:
            employees = len(data[(data["sales"] == department) & (data["salary"] == bucket)])
            left_employees = len(data[(data["sales"] == department) & (data["salary"] == bucket) & (data["left"] == 1)])
            department_salary_distribution[bucket].append(employees)
            department_salary_distribution["stack"][bucket].append(left_employees)

    draw_discrete_distribution_stacked(vars=department_salary_distribution, var_pretty_prints = buckets + ["left"],
                                       stack_pretty_print = "Left",
                                       title = "Salary per department, with left rate",
                                       ticks = departments_pretty_prints, colors=settings.large_palette)


def promotions_per_project(hr):
    projects_range = range(min(list(hr.data["number_project"])), max(list(hr.data["number_project"])) + 1)
    promotions_per_project = {}

    for bucket in [1, 0]:
        promotions_per_project[bucket] = []
        for nr_project in projects_range:
            promotions_per_project[bucket].append(
                data[(data["promotion_last_5years"] == bucket) &
                     (data["number_project"] == nr_project)]["number_project"].shape[0])

    draw_discrete_distribution_stacked(vars = promotions_per_project,
                                       var_pretty_prints = list(categorical_labels_pretty_prints["promotion_last_5years"]),
                                       title = "Promotion according to number of projects",
                                       ticks = list(map(str, projects_range)),
                                       colors = settings.large_palette_stacked)

    pp.title("Time spent in company per number of projects")
    pp.savefig("Time spent company per number of projects.png")
    pp.draw()

def time_company_per_project(hr):
    projects_range = range(min(list(hr.data["number_project"])), max(list(hr.data["number_project"])) + 1)
    time_company_per_project = {}
    time_company_buckets = 3
    cat_years = hr.data["time_spend_company"]
    cat_time = []

    for c in range(0, len(cat_years)):

        if cat_years[c] < 3:
            cat_time.append("few")
        elif cat_years[c] < 7:
            cat_time.append("medium")
        else:
            cat_time.append("high")        

    for bucket in range(time_company_buckets):
        time_company_per_project[bucket] = []
        for nr_project in projects_range:
            time_company_per_project[bucket].append(
                data[(cat_time == bucket) &
                     (data["number_project"] == nr_project)]["number_project"].shape[0])

    draw_discrete_distribution_stacked(vars = time_company_per_project,
                                       var_pretty_prints = list(categorical_labels_pretty_prints["time_spend_company"]),
                                       title = "time in company according to number of projects",
                                       ticks = list(map(str, projects_range)),
                                       colors = settings.large_palette_stacked)


def draw_discrete_log_transformation(hr, var, var_pretty_print ):
    """
    Draw the logarithm transformation for discrete values.
    :param hr: 					The data object.
    :param var: 				The variable whose log transformation to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :return: 					The data transformated.
    """
    to_log_tsc = []

    for i in range(len(hr.data[var])):
        to_log_tsc.append((np.log(hr.data[var][i])*0.54)+0.32)

    var_val_std_tsc = np.std(to_log_tsc)
    var_val_mean_tsc = np.mean(to_log_tsc)
    num_bins_tsc = int(np.ceil(np.log2(len(to_log_tsc))) + 1)

    figure, axes = pp.subplots()

    n, bins, patches = axes.hist(to_log_tsc, num_bins_tsc, label='log_transf', color=palette['main'], normed=False,
                                    stacked=True)
    y = mlab.normpdf(bins, var_val_mean_tsc, var_val_std_tsc) * sum(n * np.diff(bins))

    axes.plot(bins, y, '.-.', color=palette['secondary'], label="Gaussian approximation")
    axes.set_xlabel(str(var_pretty_print))
    axes.set_ylabel('Employees')
    axes.set_title(r'Log distribution'+str(var_pretty_print))
    pp.tight_layout()
    pp.savefig(str(var_pretty_print)+'log_distr.png')
    return to_log_tsc

def line_plot_before(values_before, var_pretty_print):
    """
    Draw the variables' line plot (values and range) before the log transformation.
    :param hr: 					The data object.
    :param var: 				The variable whose line plot to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :return: 					Nothing.
    """
    to_sort_old = values_before
    old_sorted = to_sort_old.sort_values(ascending=False)
    figure, axes = pp.subplots()
    axes.plot(range(0, len(old_sorted)), old_sorted)
    axes.set_xlabel(str(var_pretty_print))
    axes.set_ylabel('range')
    pp.savefig(str(var_pretty_print)+'Log plot before.png')
def line_plot_after(values_after, var_pretty_print):
    """
    Draw the variables' line plot (values and range) after the log transformation.
    :param hr: 					The data object.
    :param var: 				The variable whose line plot to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :return: 					Nothing.
    """    
    new_sorted = values_after 
    new_sorted.sort(reverse=True)
    figure, axes = pp.subplots()
    axes.plot(range(0, len(new_sorted)), new_sorted, '-')
    axes.set_xlabel(str(var_pretty_print))
    axes.set_ylabel('range')
    pp.savefig(str(var_pretty_print)+'Log plot after.png')


    
