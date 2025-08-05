import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


# --- Matplotlib Plotting Functions ---

def plt_plot_song_groups(score_groups, axis_name = "Sentiment Axis [-1, 1]"):
    
    direction = 0
    
    plt.figure(figsize=(10, 0.5 * sum(len(d[0]) for d in score_groups)))


    for score_dict, color, label in score_groups:

        songs = list(score_dict.keys())
        scores = list(score_dict.values())

        y_positions = [y for y in range(len(songs))]

        plt.scatter(scores, y_positions, color = color, label = label)

        if len(score_groups) > 1:
            direction = 0.1 if label.lower() == 'happy' else -0.1

        for i, song in enumerate(songs):
            ha = 'left' if scores[i] < 0 else 'right'
            plt.text(scores[i], y_positions[i] + direction, song, va='center', ha=ha, fontsize=8)



    
        

    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.yticks([])
    plt.xlabel(axis_name)
    plt.title("Songs Positioned on 1D Sentiment Axis")
    plt.grid(True, axis='x', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plt_plot_song_groups_2D(score_groups, axis_names = ('Axis 1', 'Axis 2')):




    plt.figure(figsize = (10, 8))

    for score_dict, color, label in score_groups:
        songs = list(score_dict.keys())
        scores = list(score_dict.values())

        x_scores, y_scores = zip(*scores)

        plt.scatter(x_scores, y_scores, color = color, label = label, alpha = 0.8)


        for i, song in enumerate(songs):
            plt.text(
                x_scores[i], y_scores[i] + 0.01, song,
                fontsize = 8, ha = 'center', va = 'bottom'
            )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    # Labels, grid, legend
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.title("Songs Positioned on 2D Sentiment Plane")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Plotly Plotting Functions ---

def plot_song_groups_2D(score_groups, axis_names=("Axis 1", "Axis 2")):
    all_rows = []
    
    for score_dict, color, label in score_groups:
        for song, (x, y) in score_dict.items():
            all_rows.append({
                "Song": song,
                "X": x,
                "Y": y,
                "Label": label,
                "Color": color
            })
    
    df = pd.DataFrame(all_rows)

    fig = px.scatter(
        df,
        x="X",
        y="Y",
        color="Label",
        hover_name="Song",
        color_discrete_sequence=df["Color"].unique(),
        labels={"X": axis_names[0], "Y": axis_names[1]},
        title="Songs on 2D Semantic Plane"
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        width=900,
        height=700,
        legend_title_text="Group",
        plot_bgcolor="#f9f9f9"
    )

    return fig