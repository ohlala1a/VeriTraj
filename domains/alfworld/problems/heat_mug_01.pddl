; family: heat
; task: put a hot mug in cabinet 1.
(define (problem heat-mug-01)
  (:domain veritraj-alfworld)
  (:objects
    room_1 - room
    cabinet_1 cabinet_2 countertop_1 fridge_1 garbagecan_1 microwave_1 sinkbasin_1 stoveburner_1 - receptacle
    apple_1 egg_1 knife_1 mug_1 plate_1 potato_1 - item)
  (:init
    (at agent room_1) (handempty)
    (visible cabinet_1) (visible cabinet_2) (visible countertop_1) (visible fridge_1)
    (visible garbagecan_1) (visible microwave_1) (visible sinkbasin_1) (visible stoveburner_1)
    (openable cabinet_1) (openable cabinet_2) (openable fridge_1) (openable microwave_1)
    (closed cabinet_1) (closed cabinet_2) (closed fridge_1) (closed microwave_1)
    (open countertop_1) (open garbagecan_1) (open sinkbasin_1) (open stoveburner_1)
    (can-heat microwave_1) (can-cool fridge_1) (can-clean sinkbasin_1)
    (in mug_1 countertop_1) (in knife_1 countertop_1) (in apple_1 fridge_1) (in egg_1 fridge_1)
    (in plate_1 cabinet_1) (in potato_1 garbagecan_1))
  (:goal (and (heated mug_1) (in mug_1 cabinet_1))))
