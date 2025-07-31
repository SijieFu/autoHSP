"""
Hansen Solubility Parameters related functions
"""

import os
from io import BytesIO
import base64

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import rdMolDraw2D

import streamlit as st
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import HoverTool
import plotly.graph_objects as go

from .info import info


def _add_text_to_svg(
    svg: str,
    text: str,
    x: int | str,
    y: int | str,
    font_size=12,
    font_family="Arial",
    fill="red",
):
    """
    Add text to SVG
    """
    svg = svg.strip()
    assert (
        svg[-6:] == "</svg>"
    ), f"Invalid SVG format: {svg[-6:]}. The last 6 characters should be '</svg>'"
    add = f'<text x="{x}" y="{y}" font-size="{font_size}" font-family="{font_family}" fill="{fill}">{text}</text>'
    svg = f"{svg[:-6]}\n{add}\n</svg>"
    return svg


@st.cache_data
def smiles_to_imgstr(smiles: str | dict, names: str | list, size: int = 300):
    """
    Convert ONE SMILES to image string
    """
    if isinstance(smiles, dict):
        total_vol = sum(smiles.values())
        for key in smiles:
            smiles[key] = int(smiles[key] / total_vol * 100)
    elif isinstance(smiles, str):
        smiles = {smiles: 100}
    else:
        raise ValueError(f"Invalid input type for SMILES: {type(smiles)}")

    if isinstance(names, str):
        names = [names]

    assert len(smiles) == len(
        names
    ), f"Length of SMILES and names do not match: {len(smiles)} != {len(names)}"
    rxn_smiles = []
    invalid_smi_replacement = "[He]"
    annotations = []

    for (smi, vol), name in zip(smiles.items(), names):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rxn_smiles.append(invalid_smi_replacement)
            mol = Chem.MolFromSmiles(invalid_smi_replacement)
        else:
            rxn_smiles.append(smi)
        annotations.append(f"{vol}% {name}" if len(smiles) > 1 else name)

    d2d = rdMolDraw2D.MolDraw2DSVG(size * len(smiles), size)
    opts = d2d.drawOptions()
    opts.clearBackground = False
    if len(smiles) == 1:
        d2d.DrawMolecule(mol)
    else:
        rxn_smiles = ".".join(rxn_smiles) + ">>"
        d2d.DrawReaction(Chem.ReactionFromSmarts(rxn_smiles))
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    font_family, font_size = "Arial", size // 20
    svg = _add_text_to_svg(
        svg, " + ".join(annotations), 5, size - 5, font_size, font_family, "red"
    )

    buffered = BytesIO()
    buffered.write(str.encode(svg))
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
    return img_str


def view_solvent_library():
    """
    Function to view the solvent library
    """
    st.write(f"## View Solvent Library")

    datapath = os.path.join(info.DATA_DIR, "solvent_library.csv")
    data = pd.read_csv(datapath, index_col=False)
    # remove `Water` from the solvent library
    # data = data[data["Solvent"] != "Water"]
    data["Color"] = "blue"
    columns = ["Solvent", "SMILES", "dD", "dP", "dH", "Color"]

    # the solvents that will be used for the experiment
    exp_data = pd.read_csv(os.path.join(info.DATA_DIR, "solvents.csv"), index_col=False)
    exp_data = exp_data[exp_data["InStock"].isin({"Y", "y", "1", 1})]
    exp_data["Color"] = "red"
    exp_data = exp_data[columns]

    # merge the two dataframes
    data = pd.concat([data, exp_data], ignore_index=True, axis=0)
    data["2*dD"] = data["dD"] * 2
    data = data.drop_duplicates(subset=["SMILES"], keep="last")

    img_height = 300
    imgs = data[["Solvent", "SMILES"]].apply(
        lambda x: smiles_to_imgstr({x["SMILES"]: 100}, [x["Solvent"]], size=img_height),
        axis=1,
    )
    HSP_cols = ["dD", "dP", "dH"]
    tab1, tab2 = st.tabs(["2D Visualization", "3D Visualization"])

    source = ColumnDataSource(
        data=dict(
            dD=data["dD"],
            dP=data["dP"],
            dH=data["dH"],
            Solvent=data["Solvent"],
            SMILES=data["SMILES"],
            Color=data["Color"],
            imgs=imgs,
        )
    )
    TOOLTIPS = (
        f'<img src="@imgs" height="{200}" alt="@imgs" style="float: left; margin: 0px 0px 0px 0px;" border="0"></img><br>'
        f"<b>(dD, dP, dH)</b>: (@dD{{1.1}}, @dP{{1.1}}, @dH{{1.1}})"
    )

    with tab1:
        col1, col2 = st.columns(2)
        x_axis = col1.radio(
            "X-axis:",
            HSP_cols,
            index=0,
            key="radio_HSP_solvent_library_xaxis",
            horizontal=True,
        )
        # remove the selected x-axis from the remaining columns
        y_axis = col2.radio(
            "Y-axis:",
            [col for col in HSP_cols if col != x_axis],
            index=0,
            key="radio_HSP_solvent_library_yaxis",
            horizontal=True,
        )
        with st.spinner("Rendering the 2D molplot for the solvent library..."):
            p = figure(
                width=600,
                height=600,
                title="Solvent Library",
                active_scroll="wheel_zoom",
                x_axis_label=r"$$" + x_axis + r"\small\quad\left(\sqrt{MPa}\right)$$",
                y_axis_label=r"$$" + y_axis + r"\small\quad\left(\sqrt{MPa}\right)$$",
            )
            p.scatter(
                x=x_axis, y=y_axis, source=source, size=10, color="Color", alpha=0.8
            )
            p.add_tools(HoverTool(mode="mouse", tooltips=TOOLTIPS))
            st.bokeh_chart(p, use_container_width=False)

    with tab2:
        # rewrite with plotly
        x, x2, y, z = ["dD", "2*dD", "dP", "dH"]
        with st.spinner(
            "Rendering the 3D interactive plotly plot for the solvent library..."
        ):
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=data[x2],
                        y=data[y],
                        z=data[z],
                        customdata=data.loc[:, ["Solvent", x]],
                        mode="markers",
                        marker=dict(size=5, color=data["Color"], opacity=0.5),
                        hovertemplate=(
                            f"<b>{x2}</b>: %{{x}}<br>"
                            f"<b>{x}</b>: %{{customdata[1]}}<br>"
                            f"<b>{y}</b>: %{{y}}<br>"
                            f"<b>{z}</b>: %{{z}}<br>"
                            f"<b>Solvent</b>: %{{customdata[0]}}<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_layout(
                title="Solvent Library",
                scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z),
            )
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)
