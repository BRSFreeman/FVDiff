clear classes

% pyenv('ExecutionMode',"InProcess");

% import python classes
greenGauss = py.importlib.import_module('greenGauss');

% set up test arrays in MATLAB
x_m = linspace(-1, 1, 21);
y_m = linspace(-1, 1, 21);

[xx, yy] = meshgrid(x_m, y_m);

x_r = reshape(xx, 1, []);
y_r = reshape(yy, 1, []);

xy = [x_r' y_r'];

R = [
    cos(pi/4) sin(pi/4);
    -sin(pi/4) cos(pi/4)
];

xy = xy*R;

x = xy(:, 1);
y = xy(:, 2);

u = [x.*y -0.5*y.^2];

divU = zeros(length(x),1);

% convert arrays to numpy
xy_py = py.numpy.array(xy);
u_py = py.numpy.array(u);

% calculate unstructured mesh in python
mesh = py.scipy.spatial.Voronoi(xy_py);

% use python library to calculate divergence
div = greenGauss.gaussDiv(mesh, '2d');

du_est = div(u_py);

du = double(du_est);

