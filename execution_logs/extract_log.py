import os
import re
import argparse

def extract_log_data(start_index=1, end_index=6, base_filename="xxx", output_filename="extracted_data.log"):
    """
    Iterate through a range of log files, extract lines containing specific keywords,
    and write the results to a new file.

    Args:
        start_index (int): Starting index of log files (e.g., 1 for xxx_1.log).
        end_index (int): Ending index of log files (e.g., 6 for xxx_6.log).
        base_filename (str): Base name of log files (e.g., "xxx").
        output_filename (str): Output filename to store extracted results.
    """
    # Auto-adjust keywords based on base_filename
    # Extract method name from path (e.g., OGR, OGC, TPO, etc.)
    path_parts = base_filename.replace("\\", "/").split("/")
    method_name = path_parts[1] if len(path_parts) > 1 else base_filename

    # Select val_acc keyword based on method name
    if method_name in ["OGC", "OGR"]:
        val_keyword = "Val Acc:"
    else:
        val_keyword = "val_acc ="

    # Select token keyword based on method name (methods starting with T use different format)
    if method_name.startswith("T"):
        token_keyword = "Total tokens consumed in this batch:"
    else:
        token_keyword = "Total token usage ="

    keywords = [val_keyword, token_keyword]
    print(f"Method: {method_name}, Using keywords: {keywords}")

    # Store all extracted lines
    extracted_lines = []

    # Iterate through files
    for i in range(start_index, end_index + 1):
        filename = f"{base_filename}_{i}.log"
        print(f"--- Processing file: {filename} ---")

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_lines = []
                for line in f:
                    # Check if line contains any keyword
                    if any(keyword in line for keyword in keywords):
                        # Strip whitespace and store
                        file_lines.append(line.strip())

                # If data was extracted from file, add a filename separator
                if file_lines:
                    extracted_lines.append(f"\n======== {filename} ========")
                    extracted_lines.extend(file_lines)
                else:
                    print(f"No matching data found in {filename}.")

        except FileNotFoundError:
            print(f"Error: File {filename} not found, skipping.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Write all extracted data to output file
    if extracted_lines:
        try:
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                outfile.write('\n'.join(extracted_lines))
            print(f"\n--- Extraction complete ---")
            print(f"All extracted lines successfully written to: {output_filename}")
        except Exception as e:
            print(f"Error writing to file {output_filename}: {e}")
    else:
        print("\n--- Extraction complete ---")
        print("No matching data found in any file.")

# Run function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract log data from multiple log files.")
    parser.add_argument("base_filename", help="Base name of log files (e.g., 'Main_Results/CGN/vehicle_CoT_gpt-4o_3')")
    parser.add_argument("-s", "--start", type=int, default=1, help="Starting index of log files (default: 1)")
    parser.add_argument("-e", "--end", type=int, default=6, help="Ending index of log files (default: 6)")
    parser.add_argument("-o", "--output", default="extracted_data.log", help="Output filename (default: extracted_data.log)")

    args = parser.parse_args()

    extract_log_data(
        start_index=args.start,
        end_index=args.end,
        base_filename=args.base_filename,
        output_filename=args.output
    )
