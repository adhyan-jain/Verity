import {
  Activity,
  Compass,
  Database,
  Eye,
  Layers,
  Maximize2,
  Navigation,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Zap,
} from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export interface GraphNode3D {
  id: string;
  label: string;
  type: "account" | "cardholder" | "merchant";
  riskScore: number;
  tier: "real_ledger" | "synthetic_network";
  x?: number;
  y?: number;
  z?: number;
  amount?: number;
}

export interface GraphEdge3D {
  source: string;
  target: string;
  amount: number;
  flagged?: boolean;
  timestamp?: string;
}

// Synthetic Network Dataset (FATF Structuring & Rapid-Layering multi-hop cycle)
const SYNTHETIC_NODES: GraphNode3D[] = [
  { id: "ACC-SYN-401", label: "ACC-SYN-401 (Origin)", type: "account", riskScore: 0.88, tier: "synthetic_network", x: -80, y: 30, z: 20 },
  { id: "ACC-SYN-402", label: "ACC-SYN-402 (Intermediary)", type: "account", riskScore: 0.74, tier: "synthetic_network", x: 60, y: 50, z: -40 },
  { id: "ACC-SYN-403", label: "ACC-SYN-403 (Mule Layer)", type: "account", riskScore: 0.81, tier: "synthetic_network", x: 10, y: -70, z: 60 },
  { id: "ACC-SYN-404", label: "ACC-SYN-404 (Pass-Through)", type: "account", riskScore: 0.65, tier: "synthetic_network", x: -50, y: -40, z: -50 },
  { id: "CRD-9014", label: "CRD-9014 (Cardholder)", type: "cardholder", riskScore: 0.42, tier: "synthetic_network", x: -110, y: 80, z: -10 },
  { id: "MERCH-LUX", label: "MERCH-LUX (High Value)", type: "merchant", riskScore: 0.35, tier: "synthetic_network", x: 120, y: -20, z: -20 },
  { id: "ACC-SYN-405", label: "ACC-SYN-405 (Terminal)", type: "account", riskScore: 0.91, tier: "synthetic_network", x: 90, y: -60, z: 40 },
];

const SYNTHETIC_EDGES: GraphEdge3D[] = [
  { source: "ACC-SYN-401", target: "ACC-SYN-402", amount: 48500, flagged: true },
  { source: "ACC-SYN-402", target: "ACC-SYN-403", amount: 48000, flagged: true },
  { source: "ACC-SYN-403", target: "ACC-SYN-404", amount: 47500, flagged: true },
  { source: "ACC-SYN-404", target: "ACC-SYN-401", amount: 46900, flagged: true },
  { source: "CRD-9014", target: "ACC-SYN-401", amount: 9800, flagged: false },
  { source: "ACC-SYN-403", target: "MERCH-LUX", amount: 25000, flagged: false },
  { source: "ACC-SYN-404", target: "ACC-SYN-405", amount: 49000, flagged: true },
];

// Real Ledger Dataset (10 independent accounts per spec §8 - strictly bounded)
const REAL_LEDGER_NODES: GraphNode3D[] = [
  { id: "ACC-409000493210", label: "Acct #409000493210 (Structuring)", type: "account", riskScore: 0.84, tier: "real_ledger", x: 0, y: 10, z: 0 },
  { id: "TX-CLUST-1", label: "Smurfing Cluster 1", type: "account", riskScore: 0.72, tier: "real_ledger", x: -60, y: 40, z: 30 },
  { id: "TX-CLUST-2", label: "Smurfing Cluster 2", type: "account", riskScore: 0.69, tier: "real_ledger", x: 50, y: -40, z: 40 },
  { id: "TX-CLUST-3", label: "Smurfing Cluster 3", type: "account", riskScore: 0.78, tier: "real_ledger", x: -40, y: -50, z: -30 },
  { id: "BENIGN-SALARY", label: "Monthly Payroll Counterparty", type: "merchant", riskScore: 0.12, tier: "real_ledger", x: 70, y: 50, z: -20 },
  { id: "VENDOR-RESTORE", label: "Commercial Vendor (Restore)", type: "merchant", riskScore: 0.18, tier: "real_ledger", x: -80, y: -10, z: -50 },
];

const REAL_LEDGER_EDGES: GraphEdge3D[] = [
  { source: "ACC-409000493210", target: "TX-CLUST-1", amount: 993, flagged: true },
  { source: "ACC-409000493210", target: "TX-CLUST-2", amount: 985, flagged: true },
  { source: "ACC-409000493210", target: "TX-CLUST-3", amount: 990, flagged: true },
  { source: "BENIGN-SALARY", target: "ACC-409000493210", amount: 154000, flagged: false },
  { source: "VENDOR-RESTORE", target: "ACC-409000493210", amount: 82000, flagged: false },
];

export interface NetworkGraph3DProps {
  selectedNodeId?: string;
  selectedAccountId?: string;
  onSelectNode?: (node: GraphNode3D) => void;
  className?: string;
}

export function NetworkGraph3D({
  selectedNodeId,
  selectedAccountId,
  onSelectNode,
  className = "h-[420px]",
}: NetworkGraph3DProps) {
  const effectiveNodeId = selectedNodeId || selectedAccountId;
  const containerRef = useRef<HTMLDivElement>(null);
  const [activeTier, setActiveTier] = useState<"synthetic_network" | "real_ledger">("synthetic_network");
  const [hoveredNode, setHoveredNode] = useState<GraphNode3D | null>(null);
  const [activeNode, setActiveNode] = useState<GraphNode3D | null>(null);
  const [isDrifting, setIsDrifting] = useState(true);
  const [isFlying, setIsFlying] = useState(false);

  const nodes = activeTier === "synthetic_network" ? SYNTHETIC_NODES : REAL_LEDGER_NODES;
  const edges = activeTier === "synthetic_network" ? SYNTHETIC_EDGES : REAL_LEDGER_EDGES;

  // Three.js instances ref
  const threeRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    nodeMeshes: Map<string, THREE.Mesh>;
    edgeLines: THREE.LineSegments[];
    lightParticles: THREE.Points;
    targetCamPos: THREE.Vector3 | null;
    targetLookAt: THREE.Vector3 | null;
  } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0f12); // Near-black ops-center background
    scene.fog = new THREE.FogExp2(0x0d0f12, 0.0025);

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 40, 260);

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.replaceChildren(renderer.domElement);

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(100, 150, 100);
    scene.add(dirLight);

    const gridHelper = new THREE.GridHelper(320, 16, 0x1f242d, 0x14181f);
    gridHelper.position.y = -90;
    scene.add(gridHelper);

    // 5. Node Meshes
    const nodeMeshes = new Map<string, THREE.Mesh>();
    const nodeGeometry = new THREE.SphereGeometry(1, 24, 24);

    nodes.forEach((n) => {
      const radius = n.riskScore >= 0.75 ? 8.5 : n.riskScore >= 0.4 ? 6.5 : 5.0;
      const color = n.riskScore >= 0.75
        ? 0xe05252 // Signal Red
        : n.riskScore >= 0.4
        ? 0xf59e0b // Amber
        : 0x00b4d8; // Teal / Cleared

      const mat = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.2,
        metalness: 0.8,
        emissive: color,
        emissiveIntensity: n.riskScore >= 0.7 ? 0.4 : 0.15,
      });

      const mesh = new THREE.Mesh(nodeGeometry, mat);
      mesh.scale.set(radius, radius, radius);
      mesh.position.set(n.x || 0, n.y || 0, n.z || 0);
      mesh.userData = { node: n };

      // Halo ring for flagged nodes
      if (n.riskScore >= 0.7) {
        const ringGeo = new THREE.RingGeometry(radius * 1.3, radius * 1.5, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xe05252,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.5,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        mesh.add(ring);
      }

      scene.add(mesh);
      nodeMeshes.set(n.id, mesh);
    });

    // 6. Edges (Lines between nodes)
    const edgeLines: THREE.LineSegments[] = [];
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0x3b4252,
      transparent: true,
      opacity: 0.6,
    });
    const flaggedEdgeMat = new THREE.LineBasicMaterial({
      color: 0xe05252,
      transparent: true,
      opacity: 0.9,
    });

    edges.forEach((e) => {
      const srcNode = nodes.find((n) => n.id === e.source);
      const tgtNode = nodes.find((n) => n.id === e.target);
      if (!srcNode || !tgtNode) return;

      const points = [
        new THREE.Vector3(srcNode.x || 0, srcNode.y || 0, srcNode.z || 0),
        new THREE.Vector3(tgtNode.x || 0, tgtNode.y || 0, tgtNode.z || 0),
      ];
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geo, e.flagged ? flaggedEdgeMat : edgeMat);
      scene.add(line);
    });

    // 7. Ambient Particle Field
    const particleCount = 70;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 350;
      positions[i + 1] = (Math.random() - 0.5) * 200;
      positions[i + 2] = (Math.random() - 0.5) * 350;
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0x4c566a,
      size: 2,
      transparent: true,
      opacity: 0.4,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    threeRef.current = {
      scene,
      camera,
      renderer,
      nodeMeshes,
      edgeLines,
      lightParticles: particles,
      targetCamPos: null,
      targetLookAt: null,
    };

    // 8. Interaction handling (Raycasting & Orbit Mouse Drag)
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let isMouseDown = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    const onMouseDown = (evt: MouseEvent) => {
      isMouseDown = true;
      prevMouseX = evt.clientX;
      prevMouseY = evt.clientY;
    };

    const onMouseMove = (evt: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;

      if (isMouseDown) {
        setIsDrifting(false);
        const deltaX = evt.clientX - prevMouseX;
        const deltaY = evt.clientY - prevMouseY;
        prevMouseX = evt.clientX;
        prevMouseY = evt.clientY;

        scene.rotation.y += deltaX * 0.006;
        scene.rotation.x = Math.max(-0.6, Math.min(0.6, scene.rotation.x + deltaY * 0.006));
      } else {
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(Array.from(nodeMeshes.values()));
        const firstHit = intersects[0];
        if (firstHit) {
          const target = (firstHit.object.userData as Record<string, unknown>)["node"] as GraphNode3D | undefined;
          if (target) {
            setHoveredNode(target);
            container.style.cursor = "pointer";
          } else {
            setHoveredNode(null);
            container.style.cursor = "default";
          }
        } else {
          setHoveredNode(null);
          container.style.cursor = "default";
        }
      }
    };

    const onMouseUp = () => {
      isMouseDown = false;
    };

    const onClick = () => {
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(Array.from(nodeMeshes.values()));
      const firstHit = intersects[0];
      if (firstHit) {
        const clicked = (firstHit.object.userData as Record<string, unknown>)["node"] as GraphNode3D | undefined;
        if (clicked) {
          handleFlyToNode(clicked);
        }
      }
    };

    const onWheel = (evt: WheelEvent) => {
      evt.preventDefault();
      camera.position.z = Math.max(100, Math.min(450, camera.position.z + evt.deltaY * 0.25));
    };

    container.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    container.addEventListener("click", onClick);
    container.addEventListener("wheel", onWheel, { passive: false });

    // 9. Render Loop with Ambient Idle Drift & Fly-To LERP
    let animId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const delta = clock.getDelta();

      // Ambient drift when idle
      if (isDrifting && !isMouseDown) {
        scene.rotation.y += delta * 0.12;
      }

      // Smooth Fly-To Camera Lerp
      if (threeRef.current?.targetCamPos) {
        camera.position.lerp(threeRef.current.targetCamPos, 0.06);
        if (threeRef.current.targetLookAt) {
          camera.lookAt(threeRef.current.targetLookAt);
        }
        if (camera.position.distanceTo(threeRef.current.targetCamPos) < 2) {
          threeRef.current.targetCamPos = null;
          setIsFlying(false);
        }
      }

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      container.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("click", onClick);
      container.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
    };
  }, [activeTier]);

  const handleFlyToNode = (node: GraphNode3D) => {
    setActiveNode(node);
    if (onSelectNode) onSelectNode(node);
    if (!threeRef.current) return;

    setIsFlying(true);
    setIsDrifting(false);

    const nx = node.x || 0;
    const ny = node.y || 0;
    const nz = node.z || 0;

    threeRef.current.targetCamPos = new THREE.Vector3(nx + 10, ny + 20, nz + 90);
    threeRef.current.targetLookAt = new THREE.Vector3(nx, ny, nz);
  };

  const resetView = () => {
    if (!threeRef.current) return;
    setIsFlying(true);
    threeRef.current.targetCamPos = new THREE.Vector3(0, 40, 260);
    threeRef.current.targetLookAt = new THREE.Vector3(0, 0, 0);
    setActiveNode(null);
    setIsDrifting(true);
  };

  // Step through multi-hop animation (Simulate walk_graph tool call)
  const stepWalkGraph = () => {
    const cycleNodes = nodes.filter((n) => n.riskScore >= 0.7);
    if (cycleNodes.length === 0) return;
    const nextIdx = activeNode
      ? (cycleNodes.findIndex((n) => n.id === activeNode.id) + 1) % cycleNodes.length
      : 0;
    const target = cycleNodes[nextIdx];
    if (target) {
      handleFlyToNode(target);
    }
  };

  useEffect(() => {
    if (effectiveNodeId) {
      const match = nodes.find((n) => n.id === effectiveNodeId);
      if (match) {
        handleFlyToNode(match);
      }
    }
  }, [effectiveNodeId, activeTier]);

  return (
    <div className="relative rounded-lg border border-ink/15 bg-paper/90 overflow-hidden flex flex-col font-mono shadow-soft">
      {/* Top Floating Control Bar with Structural Tier Provenance */}
      <div className="absolute top-2.5 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Tier Provenance Switcher - Structural in the UI */}
        <div className="pointer-events-auto flex items-center rounded-md border border-ink/20 bg-panel/90 backdrop-blur p-0.5 text-xs shadow-lift">
          <button
            type="button"
            onClick={() => {
              setActiveTier("synthetic_network");
              resetView();
            }}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition ${
              activeTier === "synthetic_network"
                ? "bg-signal text-signal-foreground font-bold shadow-xs"
                : "text-muted-foreground hover:text-ink"
            }`}
          >
            <Sparkles className="size-3" />
            <span>Synthetic Network (FATF Cycles)</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTier("real_ledger");
              resetView();
            }}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition ${
              activeTier === "real_ledger"
                ? "bg-teal/20 text-teal border border-teal/30 font-bold shadow-xs"
                : "text-muted-foreground hover:text-ink"
            }`}
          >
            <Database className="size-3" />
            <span>Real Ledger (10 Independent Accounts)</span>
          </button>
        </div>

        {/* Live Camera & Animation Controls */}
        <div className="pointer-events-auto flex items-center gap-1.5 rounded-md border border-ink/20 bg-panel/90 backdrop-blur px-2 py-1 text-xs shadow-lift">
          <button
            type="button"
            onClick={stepWalkGraph}
            className="flex items-center gap-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-500 px-2 py-0.5 text-[10px] font-bold border border-amber-500/40 transition"
            title="Simulate walk_graph tool call camera hop"
          >
            <Navigation className="size-3 animate-pulse" /> Step walk_graph Hop
          </button>
          <button
            type="button"
            onClick={() => setIsDrifting((prev) => !prev)}
            className={`rounded px-1.5 py-0.5 text-[10px] border transition ${
              isDrifting ? "bg-cleared/20 text-cleared border-cleared/40" : "bg-paper text-muted-foreground border-ink/10"
            }`}
            title="Toggle idle ambient drift"
          >
            {isDrifting ? "Drift Active" : "Drift Paused"}
          </button>
          <button
            type="button"
            onClick={resetView}
            className="rounded p-1 text-muted-foreground hover:text-ink border border-ink/10 bg-paper"
            title="Reset camera orientation"
          >
            <RotateCcw className="size-3" />
          </button>
        </div>
      </div>

      {/* 3D WebGL Canvas Container */}
      <div ref={containerRef} className={`w-full ${className} select-none`} />

      {/* Bottom Floating Telemetry Bar */}
      <div className="absolute bottom-2.5 left-3 right-3 z-10 flex items-center justify-between pointer-events-none text-[10px] text-muted-foreground">
        <div className="pointer-events-auto rounded bg-panel/85 backdrop-blur px-2 py-1 border border-ink/10 flex items-center gap-2">
          <span className="flex items-center gap-1 text-signal font-bold">
            <span className="size-2 rounded-full bg-signal animate-pulse" /> Critical Risk (≥0.70)
          </span>
          <span>·</span>
          <span className="flex items-center gap-1 text-amber-500 font-bold">
            <span className="size-2 rounded-full bg-amber-500" /> Medium (0.40–0.69)
          </span>
          <span>·</span>
          <span className="flex items-center gap-1 text-teal font-bold">
            <span className="size-2 rounded-full bg-teal" /> Verified Benign
          </span>
        </div>

        {/* Selected or Hovered Node Card */}
        {(activeNode || hoveredNode) && (
          <div className="pointer-events-auto rounded-md border border-ink/20 bg-panel/95 backdrop-blur px-3 py-1.5 shadow-lift text-ink animate-fadeIn flex items-center gap-3">
            <div>
              <div className="font-bold text-xs text-ink flex items-center gap-1.5">
                {(activeNode || hoveredNode)?.label}
                <span className="rounded bg-signal/15 text-signal px-1 py-0.2 text-[9px] font-bold">
                  Risk {((activeNode || hoveredNode)?.riskScore ?? 0).toFixed(2)}
                </span>
              </div>
              <div className="text-[9px] text-muted-foreground">
                Tier: {(activeNode || hoveredNode)?.tier.replace("_", " ").toUpperCase()} · Click to lock focus
              </div>
            </div>
            <button
              type="button"
              onClick={() => handleFlyToNode(activeNode || hoveredNode!)}
              className="rounded bg-signal text-signal-foreground px-2 py-1 text-[10px] font-bold"
            >
              Fly To Node
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
