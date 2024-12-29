from prettytable import PrettyTable


def format_output_to_table(response):
    # Access the actual output from the response object
    output = response.content

    # Parse the output
    lines = output.splitlines()
    headers = ["File Name"]
    table = PrettyTable(headers)

    # Extract the file names from the numbered list
    for line in lines:
        if line.strip().startswith(tuple(str(i) for i in range(1, 10))):  # Check if the line starts with a number
            file_name = line.split('. ', 1)[1]  # Split by '. ' and get the second part
            table.add_row([file_name])

    # Return the formatted table
    return table