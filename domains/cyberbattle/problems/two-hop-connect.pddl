(define (problem two-hop-connect)
 (:domain veritraj-cyberbattle)
 (:objects node_0 node_1 node_2 - node service_0 - service
           credential_0 credential_1 - credential)
 (:init (owned node_0) (discovered node_0) (discovered node_1) (discovered node_2)
        (reachable node_0 node_1) (reachable node_1 node_2)
        (has-credential credential_0) (has-credential credential_1)
        (accepts node_1 service_0 credential_0) (accepts node_2 service_0 credential_1))
 (:goal (and (owned node_2))))
