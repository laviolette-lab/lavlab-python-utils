"""Jupyter utilities for interacting with images"""

import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
from dash import html  # type: ignore
from dash.dependencies import ALL, Input, Output, State  # type: ignore
from dash_slicer import VolumeSlicer  # type: ignore
from dash_slicer.slicer import Slider, Store  # type: ignore


class ImageSliceViewer3D:  # pylint: disable=R0903
    """
    Allows thumbing through a given volume's slices interactively.

    Parameters
    ----------
    volume : np.ndarray
        3D volume to be sliced
    show : bool, optional
        Whether to display the viewer immediately, by default True
    """

    def __init__(self, volume, show=True):
        self.app = dash.Dash(__name__, update_title=None)
        slicer = VolumeSlicer(self.app, volume)
        slicer.graph.config["scrollZoom"] = False  # pylint: disable=E1101
        self.app.layout = html.Div(
            children=[slicer.graph, slicer.slider, *slicer.stores]
        )
        if show is True:
            self.show()

    def show(self):
        """
        Allows thumbing through multiple volumes' slices interactively.
        You can

        Parameters
        ----------
        volume : np.ndarray
            3D volume to be sliced
        show : bool, optional
            Whether to display the viewer immediately, by default True
        """
        # Run the app
        self.app.run_server(debug=True)


class MultiVolumeSliceViewer:
    """
    _summary_
    """

    def __init__(self, volumes, show=True):
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None
        )
        self.volumes = volumes
        self.slicers = []

        # Verify all volumes have the same slice count
        slice_counts = [volume.shape[0] for volume in volumes]
        if len(set(slice_counts)) != 1:
            raise ValueError("All volumes must have the same number of slices")
        self.slice_max = slice_counts[0] - 1  # Max index for slices

        # Container for all slicer elements
        slicer_elements = []

        for i, volume in enumerate(volumes):
            slicer = VolumeSlicer(self.app, volume, axis=0)
            slicer.graph.config["scrollZoom"] = False  # pylint: disable=E1101
            self.slicers.append(slicer)

            setpos_store = Store(
                id={"context": "app", "scene": slicer.scene_id, "name": "setpos"}
            )

            # Append slicer elements for each volume
            slicer_elements.append(
                html.Div(
                    [
                        html.H3(f"Volume {i+1}", style={"text-align": "center"}),
                        slicer.graph,
                        html.Div(
                            slicer.slider,
                            id={"type": "slicer-slider", "index": i},
                            style={"display": "none"},
                        ),
                        setpos_store,
                        *slicer.stores,
                    ],
                    id=f"slicer-{i}",
                    style={
                        "width": f"{100 / len(volumes)}%",
                        "display": "inline-block",
                    },
                )
            )  # Adjust width based on the number of volumes

        # Create a unified slider
        unified_slider = Slider(
            id="unified-slider",
            min=0,
            max=self.slice_max,
            step=1,
            value=self.slice_max // 2,
            marks={
                i: str(i)
                for i in range(0, self.slice_max + 1, max(1, self.slice_max // 10))
            },
        )

        # Create a horizontal layout of slicers
        self.slicer_container = html.Div(
            children=slicer_elements,
            id="slicers",
            style={"display": "flex", "flex-direction": "row", "align-items": "start"},
        )

        self.app.layout = html.Div(
            children=[
                self.slicer_container,
                html.Div(
                    [unified_slider],
                    id="unified-slider-container",
                    style={"display": "none", "width": "100%", "text-align": "center"},
                ),
                self.create_toggle_switch(),
                html.Div(id="info"),
            ]
        )

        self.add_sync_feature()

        if show:
            self.show()

    def show(self):
        """
        Shows the viewer if not already displayed.
        """
        # Run the app
        self.app.run_server(debug=True)

    def create_toggle_switch(self) -> dbc.Container:
        """
        Creates the toggle for switching between unified and individual slicer views.

        Returns
        -------
        dbc.Container
            Container with button.
        """
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Toggle Switch"),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Enable", "value": 1},
                                    ],
                                    value=[],
                                    id="toggle-switch",
                                    switch=True,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    justify="center",
                    align="center",
                    style={"margin-top": "50px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="toggle-output"), width=3),
                    ],
                    justify="center",
                    align="center",
                    style={"margin-top": "20px"},
                ),
            ],
            fluid=True,
        )

    def add_sync_feature(self):
        """
        Adds the synchronization feature to the viewer.
        """

        @self.app.callback(
            Output("unified-slider-container", "style"),
            [Input("toggle-switch", "value")],
        )
        def toggle_unified_slider_visibility(enabled):
            if 1 in enabled:
                return {"width": "100%", "text-align": "center"}
            return {"display": "none"}

        @self.app.callback(
            Output({"type": "slicer-slider", "index": ALL}, "style"),
            [Input("toggle-switch", "value")],
        )
        def update_slicers_visibility(enabled):
            if 1 in enabled:
                # Hide individual sliders
                return [{"display": "none"} for _ in self.slicers]
            # Show individual sliders
            return [{"display": "block"} for _ in self.slicers]

        @self.app.callback(
            Output("toggle-output", "children"), [Input("toggle-switch", "value")]
        )
        def update_output(value):
            if 1 in value:
                return "Switch is ON"
            return "Switch is OFF"

        @self.app.callback(
            Output({"context": "app", "scene": ALL, "name": "setpos"}, "data"),
            [Input("unified-slider", "value")],
            [State("toggle-switch", "value")],
        )
        def sync_slicers_from_unified(value, toggle_value):
            if toggle_value and 1 in toggle_value:
                return [(None, None, value)] * len(self.slicers)
            return [dash.no_update] * len(self.slicers)

        @self.app.callback(
            Output("unified-slider", "value"),
            [Input(slicer.state.id, "data") for slicer in self.slicers],
            [State("toggle-switch", "value")],
        )
        def sync_unified_slider(*args):
            toggle_value = args[-1]
            if not toggle_value or 1 not in toggle_value:
                return dash.no_update

            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update

            triggered_state = next(
                (state for state in args[:-1] if state["index_changed"]), None
            )
            if triggered_state:
                return triggered_state["index"]
            return dash.no_update


# Old matplotlib-based implementation
# pretty slow generally, especially over the network
# new plotly implementation is much faster due to clientside rendering

# class ImageSliceViewer3D:
#     """Allows thumbing through a given volume's slices interactively.
#     """
#     def __init__(self, volume, figsize=(6, 6), cmap='plasma', downsample_rate=1):
#         # Downsample and convert the volume to uint8
#         self.volume = ((volume[::downsample_rate,
#  ::downsample_rate, ::downsample_rate] - volume.min()) /
#                        (volume.ptp()) * 255).astype(np.uint8)
#         self.figsize = figsize
#         self.cmap = cmap
#         self.v = [np.min(self.volume), np.max(self.volume)]

#         # Slider for selecting the slice within the selected plane
#         self.slice_slider = ipyw.IntSlider(min=0, max=self.volume.shape[2]-1, step=1,
#                                            continuous_update=True, description='Image Slice:')

#         # Initialize the plot
#         self.fig, self.ax = plt.subplots(figsize=self.figsize)
#         self.img = self.ax.imshow(self._get_slice(0), cmap=self.cmap,
#                                   vmin=self.v[0], vmax=self.v[1])
#         self.cbar = self.fig.colorbar(self.img, ax=self.ax)
#         self.ax.set_title('Slice 0')

#         # Output widget
#         self.out = ipyw.Output()
#         with self.out:
#             plt.show()

#         # Layout widgets and display
#         display(VBox([self.out, self.slice_slider]))

#         # Register event handlers
#         self.slice_slider.observe(self._update_slice, names='value')

#     def _get_slice(self, z):
#         return self.volume[:, :, z]

#     def _update_slice(self, change):
#         z = change['new']
#         with self.out:
#             self.img.set_data(self._get_slice(z))
#             self.ax.set_title(f'Slice {z}')
#             self.img.set_clim(self.v[0], self.v[1])  # Predefined color limits
#             self.fig.canvas.draw_idle()

# class MultiImageSliceViewer3D:
#     """Allows thumbing through multiple volumes' slices interactively,
# with the ability to lock the views."""

#     def __init__(self, volumes, figsize=(6, 6), cmap='plasma', downsample_rate=1):
#         self.volumes = [((vol[::downsample_rate, ::downsample_rate,
#  ::downsample_rate] - vol.min()) /
#                          vol.ptp() * 255).astype(np.uint8) for vol in volumes]
#         self.figsize = figsize
#         self.cmap = cmap
#         self.num_images = len(self.volumes)
#         self.v = [np.min([np.min(vol) for vol in self.volumes]),
# np.max([np.max(vol) for vol in self.volumes])]
#         self.lock_views = False
#         self.previous_slice_indices = [0] * self.num_images

#         self.slice_sliders = []
#         self.axs = []
#         self.imgs = []

#         self.fig, axs = plt.subplots(1, self.num_images,
# figsize=(self.figsize[0] * self.num_images, self.figsize[1]))

#         if not isinstance(axs, np.ndarray):
#             axs = [axs]

#         for i, vol in enumerate(self.volumes):
#             slider = ipyw.IntSlider(min=0, max=vol.shape[2]-1, step=1,
# continuous_update=True, description=f'Image {i+1}:')
#             self.slice_sliders.append(slider)

#             ax = axs[i]
#             self.axs.append(ax)

#             img = ax.imshow(self._get_slice(vol, 0),
# cmap=self.cmap, vmin=self.v[0], vmax=self.v[1])
#             self.imgs.append(img)
#             self.fig.colorbar(img, ax=ax)
#             ax.set_title(f'Image {i+1} - Slice 0')

#         self.out = ipyw.Output()
#         with self.out:
#             plt.show()

#         self.lock_checkbox = ipyw.Checkbox(value=False, description='Lock Views')

#         sliders_box = HBox([slider for slider in self.slice_sliders])
#         display(VBox([self.out, sliders_box, self.lock_checkbox]))

#         for slider in self.slice_sliders:
#             slider.observe(self._update_slices, names='value')

#         self.lock_checkbox.observe(self._toggle_lock, names='value')

#         # Connect events to synchronize zoom and pan
#         self.fig.canvas.mpl_connect('button_release_event', self._sync_axes)
#         self.fig.canvas.mpl_connect('motion_notify_event', self._sync_axes)

#     def _get_slice(self, volume, z):
#         return volume[:, :, z]

#     def _update_slices(self, change):
#         slider_index = self.slice_sliders.index(change['owner'])
#         new_index = change['new']
#         relative_change = new_index - self.previous_slice_indices[slider_index]
#         self.previous_slice_indices[slider_index] = new_index

#         for i, vol in enumerate(self.volumes):
#             if self.lock_views and i != slider_index:
#                 self.slice_sliders[i].unobserve(self._update_slices, names='value')
#                 new_slider_value = self.slice_sliders[i].value + relative_change
#                 new_slider_value = np.clip(new_slider_value, 0, self.slice_sliders[i].max)
#                 self.slice_sliders[i].value = new_slider_value
#                 self.slice_sliders[i].observe(self._update_slices, names='value')

#             with self.out:
#                 self.imgs[i].set_data(self._get_slice(vol, self.slice_sliders[i].value))
#                 self.axs[i].set_title(f'Image {i+1} - Slice {self.slice_sliders[i].value}')
#                 self.imgs[i].set_clim(self.v[0], self.v[1])  # Predefined color limits

#         self.fig.canvas.draw_idle()

#     def _toggle_lock(self, change):
#         self.lock_views = change['new']

#     def _sync_axes(self, event):
#         if not self.lock_views or event.inaxes is None:
#             return

#         for ax in self.axs:
#             if ax is not event.inaxes:
#                 ax.set_xlim(event.inaxes.get_xlim())
#                 ax.set_ylim(event.inaxes.get_ylim())
#         self.fig.canvas.draw_idle()
