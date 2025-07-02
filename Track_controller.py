import ctypes
import time
import xr
import json
import socket

#获取手柄的位姿数据以及按钮数据


class OpenXRController:
    def __init__(self):
        # UDP 配置
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5006
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.n = 0

    def run(self):
        with xr.ContextObject(
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
                ),
        ) as context:
            controller_paths = (xr.Path * 2)(
                xr.string_to_path(context.instance, "/user/hand/left"),
                xr.string_to_path(context.instance, "/user/hand/right"),
            )

            # 创建手柄姿态动作
            controller_pose_action = xr.create_action(
                action_set=context.default_action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.POSE_INPUT,
                    action_name="hand_pose",
                    localized_action_name="Hand Pose",
                    subaction_paths=controller_paths,
                ),
            )

            # 创建按钮动作 - 选择按钮(Select)和菜单按钮(Menu)
            select_action = xr.create_action(
                action_set=context.default_action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name="select_button",
                    localized_action_name="Select Button",
                    subaction_paths=controller_paths,
                ),
            )

            menu_action = xr.create_action(
                action_set=context.default_action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name="menu_button",
                    localized_action_name="Menu Button",
                    subaction_paths=controller_paths,
                ),
            )

            # 配置动作绑定
            suggested_bindings = (xr.ActionSuggestedBinding * 6)(
                # 姿态绑定
                xr.ActionSuggestedBinding(
                    action=controller_pose_action,
                    binding=xr.string_to_path(context.instance, "/user/hand/left/input/grip/pose"),
                ),
                xr.ActionSuggestedBinding(
                    action=controller_pose_action,
                    binding=xr.string_to_path(context.instance, "/user/hand/right/input/grip/pose"),
                ),
                # 选择按钮绑定
                xr.ActionSuggestedBinding(
                    action=select_action,
                    binding=xr.string_to_path(context.instance, "/user/hand/left/input/select/click"),
                ),
                xr.ActionSuggestedBinding(
                    action=select_action,
                    binding=xr.string_to_path(context.instance, "/user/hand/right/input/select/click"),
                ),
                # 菜单按钮绑定
                xr.ActionSuggestedBinding(
                    action=menu_action,
                    binding=xr.string_to_path(context.instance, "/user/hand/left/input/menu/click"),
                ),
                xr.ActionSuggestedBinding(
                    action=menu_action,
                    binding=xr.string_to_path(context.instance, "/user/hand/right/input/menu/click"),
                ),
            )

            xr.suggest_interaction_profile_bindings(
                instance=context.instance,
                suggested_bindings=xr.InteractionProfileSuggestedBinding(
                    interaction_profile=xr.string_to_path(context.instance, "/interaction_profiles/khr/simple_controller"),
                    count_suggested_bindings=len(suggested_bindings),
                    suggested_bindings=suggested_bindings,
                ),
            )

            action_spaces = [
                xr.create_action_space(
                    session=context.session,
                    create_info=xr.ActionSpaceCreateInfo(
                        action=controller_pose_action,
                        subaction_path=controller_paths[0],
                    ),
                ),
                xr.create_action_space(
                    session=context.session,
                    create_info=xr.ActionSpaceCreateInfo(
                        action=controller_pose_action,
                        subaction_path=controller_paths[1],
                    ),
                ),
            ]

            session_was_focused = False
            for frame_index, frame_state in enumerate(context.frame_loop()):
                self.n += 1
                if context.session_state == xr.SessionState.FOCUSED:
                    session_was_focused = True
                    active_action_set = xr.ActiveActionSet(
                        action_set=context.default_action_set,
                        subaction_path=xr.NULL_PATH,
                    )
                    xr.sync_actions(
                        session=context.session,
                        sync_info=xr.ActionsSyncInfo(
                            count_active_action_sets=1,
                            active_action_sets=ctypes.pointer(active_action_set),
                        ),
                    )

                    poses = {}
                    buttons = {"left": {}, "right": {}}

                    # 获取按钮状态
                    for i, hand_path in enumerate(controller_paths):
                        hand = "left" if i == 0 else "right"

                        # 获取选择按钮状态
                        select_state = xr.get_action_state_boolean(
                            session=context.session,
                            get_info=xr.ActionStateGetInfo(
                                action=select_action,
                                subaction_path=hand_path,
                            ),
                        )
                        if select_state.is_active:
                            buttons[hand]["select"] = select_state.current_state

                        # 获取菜单按钮状态
                        menu_state = xr.get_action_state_boolean(
                            session=context.session,
                            get_info=xr.ActionStateGetInfo(
                                action=menu_action,
                                subaction_path=hand_path,
                            ),
                        )
                        if menu_state.is_active:
                            buttons[hand]["menu"] = menu_state.current_state

                    # 获取手柄姿态
                    for index, space in enumerate(action_spaces):
                        space_location = xr.locate_space(
                            space=space,
                            base_space=context.space,
                            time=frame_state.predicted_display_time,
                        )
                        hand = "left" if index == 0 else "right"
                        if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                            pose = {
                                "position": {
                                    "x": space_location.pose.position.x,
                                    "y": space_location.pose.position.y,
                                    "z": space_location.pose.position.z,
                                },
                                "orientation": {
                                    "x": space_location.pose.orientation.x,
                                    "y": space_location.pose.orientation.y,
                                    "z": space_location.pose.orientation.z,
                                    "w": space_location.pose.orientation.w,
                                },
                            }
                            poses[hand] = pose

                    # 发送姿态和按钮数据
                    if self.n >= 10 and (poses or buttons):
                        data = {
                            "info": poses,
                            "buttons": buttons
                        }
                        self.sock.sendto(json.dumps(data).encode(), (self.UDP_IP, self.UDP_PORT))

                time.sleep(0.01)  # 50Hz 更新频率
                if frame_index > 1000000000:  # 长时间运行
                    break

            if not session_was_focused:
                print("OpenXR session not focused. Is the headset on?")
            self.sock.close()  # 关闭 Socket

# 外部调用示例
if __name__ == "__main__":
    controller = OpenXRController()
    controller.run()