from src.utils.common_utils import log



def target_encoding(data, xcol, dct_map, log_file):
    """
        It helps to apply target_encoding.
        :param data: data.csv
        :param col: column
        :param dct_map: mapping dictionary
        :param log_file: log_file.txt
        :return: data
    """
    try:
        data = data
        file = log_file
        log(file_object=file, log_message=f"perform target encoding encoding on xcol: {xcol}")  # logs the details

        data[xcol] = data[xcol].map(dct_map)
        return data

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e} \n\n")  # logs the details
        raise e


def mean_encoding(data, xcol, ycol, log_file):
    """
        It helps to apply mean_encoding.\n
        :param data: data
        :param xcol: xcol
        :param ycol: ycol
        :param log_file: log_file.txt
        :return: data, dct
    """
    try:
        data = data
        file = log_file
        log(file_object=file, log_message=f"perform Mean encoding on xcol: {xcol}, ycol: {ycol}")  # logs the details

        dct = data.groupby([xcol])[ycol].mean().sort_values(ascending=False).to_dict() # get dictionary
        data = target_encoding(data=data, xcol=xcol, dct_map=dct, log_file=log_file)
        return data, dct


    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e} \n\n")  # logs the details
        raise e



if __name__ == '__main__':
    pass

