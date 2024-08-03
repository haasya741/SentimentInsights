import pandas as pd
import sys

from similarity.create_embeddings import preprocess_data
from similarity.query_embeddings import query_embeddings


def generate_output_file(
    partner_file, internal_file, mapping_file, input_file, output_file
):
    # Load data
    partner_data = pd.read_csv(partner_file)
    internal_data = pd.read_csv(internal_file)
    internal_data = preprocess_data(internal_data, "description")
    mapping_data = pd.read_csv(mapping_file)
    input_data = pd.read_csv(input_file)

    # Merge mapping data to get internal names and IDs
    # merged_data = input_data.merge(mapping_data, how='left', left_on='item_name', right_on='partner_name')
    merged_data = input_data.merge(
        mapping_data, how="left", left_on="row_num", right_on="external_row_num"
    ).merge(internal_data, how="left", left_on="internal_row_num", right_on="row_num")
    # print(merged_data.describe())
    print(merged_data.head())
    # return
    # Output list to store results
    output_list = []

    # Iterate through each input item
    for _, row in input_data.iterrows():
        partner_name = row["description"].lower()
        mapped_row = merged_data[merged_data["description_y"] == partner_name]

        if not mapped_row.empty and pd.notnull(mapped_row["description_y"].values[0]):
            internal_name = mapped_row["description_y"].values[0]
            internal_id = mapped_row["row_num_y"].values[0]
            output_list.append((partner_name, internal_id, 1))
        else:
            similar_items = query_embeddings(
                partner_name, top_k=3, model_path="./cpg_index"
            )
            for similar_item in similar_items:
                internal_name, score = similar_item
                normalized_score = 1 - score.item() / 2
                # print(internal_name.page_content, score)
                internal_id = internal_data[
                    internal_data["description"] == internal_name.page_content
                ]["row_num"].values[0]
                output_list.append(
                    (
                        partner_name,
                        internal_id,
                        internal_name.page_content,
                        normalized_score,
                    )
                )
                if score == 1:
                    break

    # Create a DataFrame for the output
    output_df = pd.DataFrame(
        output_list, columns=["partner_name", "id", "internal_name", "score"]
    )

    # Save to CSV
    output_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Usage: python script.py <partner_file> <internal_file> <mapping_file> <input_file> <output_file>"
        )
        sys.exit(1)

    partner_file = sys.argv[1]
    internal_file = sys.argv[2]
    mapping_file = sys.argv[3]
    input_file = sys.argv[4]
    output_file = sys.argv[5]

    generate_output_file(
        partner_file, internal_file, mapping_file, input_file, output_file
    )
