import logging

from prettytable import PrettyTable, ALL

logger = logging.getLogger(__name__)


def gen_table(run_name, metrics, process_type, bias_type):
    try:

        res_table = PrettyTable(hrules=ALL, vrules=ALL, border=True)

        res_table.title = str(run_name)
        res_table.field_names = ["Parameter", "Value"]
        res_table.add_row(["Process Type", str(process_type)])
        res_table.add_row(["Bias Type", str(bias_type)])

        for key, val in metrics.items():
            res_table.add_row([str(key).capitalize(), str(round(val, 4))])

        return res_table

    except Exception as e:
        logger.error(e)
