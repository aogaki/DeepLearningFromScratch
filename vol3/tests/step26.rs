use vol3::utils::get_dot_graph;
use vol3::variable::Variable;

#[test]
fn test_get_dot_graph() {
    let x0 = Variable::from(1.0);
    x0.set_name("x0");

    let x1 = Variable::from(1.0);
    x1.set_name("x1");

    let y = &x0 + &x1;
    y.set_name("y");

    let txt = get_dot_graph(&y, false);

    // Assert it contains the digraph boilerplate
    assert!(txt.contains("digraph g {"));
    assert!(txt.contains("}"));

    // Assert it contains the nodes and the function (Add)
    assert!(txt.contains("[label=\"x0\", color=orange, style=filled]"));
    assert!(txt.contains("[label=\"x1\", color=orange, style=filled]"));
    assert!(txt.contains("[label=\"y\", color=orange, style=filled]"));
    assert!(txt.contains("[label=\"Add\", color=lightblue, style=filled, shape=box]"));

    // Assert it contains edges (->)
    assert!(txt.contains("->"));
}
