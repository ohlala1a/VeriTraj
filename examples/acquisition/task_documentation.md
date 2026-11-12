# Household environment: task documentation (input d to the Builder)

The agent acts in a single room with numbered receptacles (cabinet 1, countertop 1, fridge 1, ...)
and portable objects (mug 1, apple 2, ...). Commands:

- `go to <receptacle>`: moves the agent to a visible receptacle.
- `open <receptacle>` / `close <receptacle>`: only cabinets, drawers, fridges, microwaves and safes
  can be opened or closed. Other receptacles (countertops, tables, shelves, ...) are always accessible.
- `take <object> from <receptacle>`: the agent must be at the receptacle, its hand must be empty, and
  the receptacle must be accessible (open, if it can be closed).
- `put <object> in/on <receptacle>`: the agent must be at the receptacle and hold the object; closed
  receptacles do not accept objects.
- `heat <object> with microwave`, `cool <object> with fridge`, `clean <object> with sinkbasin`: the agent
  must hold the object and stand at the appliance.
- `use <lamp>`: turns on a lamp standing on the receptacle where the agent is.
- `examine <object> with <lamp>`: the agent holds the object, stands where the lamp is, and the lamp is on.
- `look`, `inventory`: no effect on the world.

If a command cannot be carried out the environment answers "Nothing happens." and nothing changes.
