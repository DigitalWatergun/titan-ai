from rich.console import RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.geometry import Offset, Region
from textual.screen import Screen
from textual.selection import SelectEnd, Selection, SelectStart, SelectState
from textual.widget import Widget
from textual.widgets import Markdown, Static


class CopyCaret(Static):
    ALLOW_SELECT = False
    DEFAULT_CSS = """
    CopyCaret {
        layer: caret;
        width: 1;
        height: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__(" ", markup=False)


class ChatLog(VerticalScroll, can_focus=True):
    DEFAULT_CSS = """
    ChatLog {
        layers: base caret;
    }
    """

    BINDINGS = [
        ("j", "cursor_down", ""),
        ("k", "cursor_up", ""),
        ("h", "cursor_left", ""),
        ("l", "cursor_right", ""),
        ("g", "cursor_top", ""),
        ("G", "cursor_bottom", ""),
        ("ctrl+d", "cursor_half_down", ""),
        ("ctrl+u", "cursor_half_up", ""),
        ("home", "cursor_line_start", ""),
        ("end", "cursor_line_end", ""),
        ("0", "cursor_line_start", ""),
        ("$", "cursor_line_end", ""),
        ("v", "toggle_visual", ""),
        ("y", "yank", ""),
        ("escape", "leave_copy_mode", ""),
        ("q", "leave_copy_mode", ""),
        ("i", "leave_copy_mode", ""),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._copy_mode = False
        self._visual = False
        self._cursor_row = 0
        self._cursor_col = 0
        self._desired_col = 0
        self._caret: CopyCaret | None = None
        self._anchor_pos: tuple[int, int] | None = None
        self._anchor_forward: tuple[Widget, Offset, int, int] | None = None
        self._anchor_backward: tuple[Widget, Offset, int, int] | None = None

    def _screen_offset(self, row: int, col: int) -> Offset:
        scroll_x, scroll_y = self.scroll_offset
        region = self.content_region
        return Offset(region.x + col - scroll_x, region.y + row - scroll_y)

    def _scroll_row_into_view(self, row: int) -> None:
        _, scroll_y = self.scroll_offset
        height = self.content_region.height
        if row < scroll_y:
            self.scroll_to(y=row, animate=False, immediate=True)
            self.screen._refresh_layout()
        elif row >= scroll_y + height:
            self.scroll_to(y=row - height + 1, animate=False, immediate=True)
            self.screen._refresh_layout()

    def _probe_cell(self, offset: Offset) -> tuple[Widget, Offset] | None:
        widget, content_offset = self.screen.get_widget_and_offset_at(
            offset.x, offset.y
        )
        if widget is None or content_offset is None:
            return None
        if widget.children or isinstance(widget, CopyCaret):
            return None
        if not widget.content_region.contains(offset.x, offset.y):
            return None
        character = widget.get_selection(
            Selection.from_offsets(content_offset, content_offset + Offset(1, 0))
        )
        if character is None or not character[0]:
            return None
        return widget, content_offset

    def _row_text_extent(self, row: int) -> tuple[int, int] | None:
        region = self.content_region
        probe_y = region.y + row - self.scroll_offset.y
        if not region.y <= probe_y < region.bottom:
            return None
        character_cols = [
            col
            for col in range(region.width)
            if self._probe_cell(self._screen_offset(row, col)) is not None
        ]
        if not character_cols:
            return None
        return character_cols[0], character_cols[-1]

    def _find_text_row(self, start_row: int, direction: int) -> int | None:
        last_row = max(0, self.virtual_size.height - 1)
        row = start_row
        while 0 <= row <= last_row:
            if self._row_text_extent(row) is not None:
                return row
            row += direction
        return None

    def _move_cursor_row(self, target_row: int) -> None:
        last_row = max(0, self.virtual_size.height - 1)
        row = max(0, min(target_row, last_row))
        self._scroll_row_into_view(row)
        extent = self._row_text_extent(row)
        self._cursor_row = row
        if extent is None:
            self._cursor_col = 0
        else:
            first_col, last_col = extent
            self._cursor_col = min(max(self._desired_col, first_col), last_col)
        self._paint_cursor()

    def _move_to_last_text_line(self) -> None:
        target_y = max(0, self.virtual_size.height - self.content_region.height)
        if self.scroll_offset.y != target_y:
            self.scroll_to(y=target_y, animate=False, immediate=True)
            self.screen._refresh_layout()
        last_row = max(0, self.virtual_size.height - 1)
        last_text_row = self._find_text_row(last_row, -1)
        self._move_cursor_row(last_row if last_text_row is None else last_text_row)

    def _move_below_last_text_line(self) -> None:
        target_y = max(0, self.virtual_size.height - self.content_region.height)
        if self.scroll_offset.y != target_y:
            self.scroll_to(y=target_y, animate=False, immediate=True)
            self.screen._refresh_layout()
        last_row = max(0, self.virtual_size.height - 1)
        last_text_row = self._find_text_row(last_row, -1)
        if last_text_row is None:
            self._move_cursor_row(last_row)
        else:
            self._move_cursor_row(min(last_text_row + 1, last_row))

    def _select_container(self, widget: Widget) -> Widget:
        if isinstance(widget, Screen):
            return widget
        parent = widget.parent
        if isinstance(parent, Widget):
            return parent
        return self.screen

    def _anchor_at_cursor(self) -> None:
        offset = self._screen_offset(self._cursor_row, self._cursor_col)
        probed = self._probe_cell(offset)
        if probed is None:
            self.screen.clear_selection()
            return
        widget, content_offset = probed
        container = self._select_container(widget)
        anchor = SelectState(
            offset,
            start=SelectStart(
                container,
                offset - container.region.offset,
                container.region.offset,
                container.scroll_offset,
                content_widget=widget,
                content_offset=content_offset,
            ),
        )
        self.screen._select_state = anchor.update_end(
            offset, SelectEnd(container, widget, content_offset)
        )

    def _resolve_text(
        self, row: int, col: int, direction: int
    ) -> tuple[Widget, Offset, int, int] | None:
        last_row = max(0, self.virtual_size.height - 1)
        current_row = row
        while 0 <= current_row <= last_row:
            extent = self._row_text_extent(current_row)
            if extent is not None:
                first_col, last_col = extent
                if direction > 0:
                    current_col = (
                        max(col, first_col) if current_row == row else first_col
                    )
                else:
                    current_col = min(col, last_col) if current_row == row else last_col
                while first_col <= current_col <= last_col:
                    probed = self._probe_cell(
                        self._screen_offset(current_row, current_col)
                    )
                    if probed is not None:
                        return (probed[0], probed[1], current_row, current_col)
                    current_col += direction
            current_row += direction
        return None

    def _extend_selection(self) -> None:
        if self._anchor_pos is None:
            return
        cursor_pos = (self._cursor_row, self._cursor_col)
        if cursor_pos >= self._anchor_pos:
            start = self._anchor_forward
            end = self._resolve_text(self._cursor_row, self._cursor_col, -1)
        else:
            start = self._resolve_text(self._cursor_row, self._cursor_col, 1)
            end = self._anchor_backward
        if start is None or end is None:
            self.screen.clear_selection()
            return
        start_widget, start_content, start_row, start_col = start
        end_widget, end_content, end_row, end_col = end
        if (start_row, start_col) > (end_row, end_col):
            self.screen.clear_selection()
            return
        if not start_widget.is_attached or not end_widget.is_attached:
            return
        if start_widget is not end_widget:
            end_content = end_content + Offset(1, 0)
        start_screen = self._screen_offset(start_row, start_col)
        end_screen = self._screen_offset(end_row, end_col)
        container = self._select_container(start_widget)
        self.screen._select_state = SelectState(
            end_screen,
            start=SelectStart(
                container,
                start_screen - container.region.offset,
                container.region.offset,
                container.scroll_offset,
                content_widget=start_widget,
                content_offset=start_content,
            ),
            end=SelectEnd(self._select_container(end_widget), end_widget, end_content),
        )

    def _paint_cursor(self) -> None:
        self._scroll_row_into_view(self._cursor_row)
        cursor_on_text = self._row_text_extent(self._cursor_row) is not None
        if self._caret is not None:
            self._caret.display = not cursor_on_text
            if not cursor_on_text:
                self._caret.styles.offset = (self._cursor_col, self._cursor_row)
        if self._visual:
            self._extend_selection()
        elif cursor_on_text:
            self._anchor_at_cursor()
        else:
            self.screen.clear_selection()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "leave_copy_mode":
            return True
        if action == "yank":
            return self._copy_mode or bool(self.screen.selections)
        return self._copy_mode

    def enter_copy_mode(self) -> None:
        self._copy_mode = True
        self._visual = False
        self._desired_col = 0
        self._caret = CopyCaret()
        self.mount(self._caret)
        self.focus()
        self._move_below_last_text_line()

    def leave_copy_mode(self) -> None:
        self._copy_mode = False
        self._visual = False
        if self._caret is not None:
            self._caret.remove()
            self._caret = None
        self._anchor_pos = None
        self._anchor_forward = None
        self._anchor_backward = None
        self.screen.clear_selection()
        self.app.query_one("#input-box").focus()

    def action_leave_copy_mode(self) -> None:
        self.leave_copy_mode()

    def action_cursor_down(self) -> None:
        self._move_cursor_row(self._cursor_row + 1)

    def action_cursor_up(self) -> None:
        self._move_cursor_row(self._cursor_row - 1)

    def action_cursor_left(self) -> None:
        extent = self._row_text_extent(self._cursor_row)
        if extent is not None:
            self._cursor_col = max(self._cursor_col - 1, extent[0])
            self._desired_col = self._cursor_col
        self._paint_cursor()

    def action_cursor_right(self) -> None:
        extent = self._row_text_extent(self._cursor_row)
        if extent is not None:
            self._cursor_col = min(self._cursor_col + 1, extent[1])
            self._desired_col = self._cursor_col
        self._paint_cursor()

    def action_cursor_top(self) -> None:
        if self.scroll_offset.y != 0:
            self.scroll_to(y=0, animate=False, immediate=True)
            self.screen._refresh_layout()
        first_text_row = self._find_text_row(0, 1)
        self._move_cursor_row(0 if first_text_row is None else first_text_row)

    def action_cursor_bottom(self) -> None:
        self._move_to_last_text_line()

    def action_cursor_half_down(self) -> None:
        self._move_cursor_row(self._cursor_row + self.content_region.height // 2)

    def action_cursor_half_up(self) -> None:
        self._move_cursor_row(self._cursor_row - self.content_region.height // 2)

    def action_cursor_line_start(self) -> None:
        extent = self._row_text_extent(self._cursor_row)
        if extent is not None:
            self._cursor_col = extent[0]
            self._desired_col = self._cursor_col
        self._paint_cursor()

    def action_cursor_line_end(self) -> None:
        extent = self._row_text_extent(self._cursor_row)
        if extent is not None:
            self._cursor_col = extent[1]
            self._desired_col = self._cursor_col
        self._paint_cursor()

    def action_toggle_visual(self) -> None:
        self._visual = not self._visual
        self.screen.clear_selection()
        if self._visual:
            self._anchor_pos = (self._cursor_row, self._cursor_col)
            self._anchor_forward = self._resolve_text(
                self._cursor_row, self._cursor_col, 1
            )
            self._anchor_backward = self._resolve_text(
                self._cursor_row, self._cursor_col, -1
            )
        else:
            self._anchor_pos = None
            self._anchor_forward = None
            self._anchor_backward = None
        self._paint_cursor()

    def _selection_text(self) -> str:
        fragments: list[tuple[Region, str, str]] = []
        for widget, selection in self.screen.selections.items():
            if not widget.is_attached:
                continue
            extracted = widget.get_selection(selection)
            if extracted is None or not extracted[0].strip():
                continue
            text, ending = extracted
            fragments.append((widget.content_region, text, ending))
        fragments.sort(key=lambda fragment: (fragment[0].y, fragment[0].x))
        parts: list[str] = []
        previous_region: Region | None = None
        previous_ending = ""
        for region, text, ending in fragments:
            if previous_region is not None:
                blank_rows = region.y - previous_region.bottom
                parts.append(
                    previous_ending if blank_rows < 0 else "\n" * (1 + blank_rows)
                )
            parts.append(text)
            previous_region = region
            previous_ending = ending
        return "".join(parts)

    def action_yank(self) -> None:
        selected_text = self._selection_text().rstrip("\n")
        if selected_text:
            self.app.copy_to_clipboard(selected_text)
        self.leave_copy_mode()

    def write(self, renderable: RenderableType) -> None:
        if isinstance(renderable, str):
            renderable = Text.from_markup(renderable)
        self.mount(Static(renderable, markup=False))
        self.call_after_refresh(self.scroll_end, animate=False)

    def write_markdown(self, content: str) -> None:
        self.mount(Markdown(content))
        self.call_after_refresh(self.scroll_end, animate=False)

    def clear(self) -> None:
        self.remove_children()
