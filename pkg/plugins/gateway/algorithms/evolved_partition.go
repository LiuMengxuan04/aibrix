/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package routingalgorithms

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

const RouterEvolvedPartition types.RoutingAlgorithm = "evolved-partition"
const RouterEvolvedTokenBalance types.RoutingAlgorithm = "evolved-token-balance"
const RouterEvolvedSILV2 types.RoutingAlgorithm = "evolved-sil-v2"
const RouterEvolvedOutstanding types.RoutingAlgorithm = "evolved-outstanding"
const RouterE2EPortableAgent types.RoutingAlgorithm = "e2e-portable-agent"
const RouterE2EGPUPartition types.RoutingAlgorithm = "e2e-gpu-partition"
const RouterSILV6CV2 types.RoutingAlgorithm = "sil-v6c-v2"
const RouterCleanSILV3AgentB types.RoutingAlgorithm = "clean-sil-v3-agent-b"
const RouterCleanSILV8AgentK types.RoutingAlgorithm = "clean-sil-v8-agent-k"
const RouterCleanSILV8AgentS types.RoutingAlgorithm = "clean-sil-v8-agent-s"
const RouterCleanSILV9AgentV types.RoutingAlgorithm = "clean-sil-v9-agent-v"
const RouterCleanSILV10AgentY types.RoutingAlgorithm = "clean-sil-v10-agent-y"
const RouterCleanSILV11AgentZ types.RoutingAlgorithm = "clean-sil-v11-agent-z"
const RouterCleanSILV12AgentM types.RoutingAlgorithm = "clean-sil-v12-agent-m"
const RouterCleanSILV12AgentMToken types.RoutingAlgorithm = "clean-sil-v12-agent-m-token"
const RouterCleanSILV12AgentL types.RoutingAlgorithm = "clean-sil-v12-agent-l"
const RouterCleanSILV13AgentN types.RoutingAlgorithm = "clean-sil-v13-agent-n"
const RouterCleanSILV13AgentP types.RoutingAlgorithm = "clean-sil-v13-agent-p"
const RouterCleanSILV13AgentPTight types.RoutingAlgorithm = "clean-sil-v13-agent-p-tight"
const evolvedPartitionTenantHeader = "x-sil-tenant"
const evolvedPartitionTraceRequestHeader = "x-sil-trace-request-id"

type evolvedPartitionTrackedKeyStruct struct{}

var evolvedPartitionTrackedKey = evolvedPartitionTrackedKeyStruct{}

func init() {
	Register(RouterEvolvedPartition, NewEvolvedPartitionRouter)
	Register(RouterEvolvedTokenBalance, NewEvolvedPartitionRouter)
	Register(RouterEvolvedSILV2, NewEvolvedSILV2Router)
	Register(RouterEvolvedOutstanding, NewEvolvedOutstandingRouter)
	Register(RouterE2EPortableAgent, NewE2EPortableAgentRouter)
	Register(RouterE2EGPUPartition, NewE2EGPUPartitionRouter)
	Register(RouterSILV6CV2, NewSILV6CV2Router)
	Register(RouterCleanSILV3AgentB, NewCleanSILV3AgentBRouter)
	Register(RouterCleanSILV8AgentK, NewCleanSILV8AgentKRouter)
	Register(RouterCleanSILV8AgentS, NewCleanSILV8AgentSRouter)
	Register(RouterCleanSILV9AgentV, NewCleanSILV9AgentVRouter)
	Register(RouterCleanSILV10AgentY, NewCleanSILV10AgentYRouter)
	Register(RouterCleanSILV11AgentZ, NewCleanSILV11AgentZRouter)
	Register(RouterCleanSILV12AgentM, NewCleanSILV12AgentMRouter)
	Register(RouterCleanSILV12AgentMToken, NewCleanSILV12AgentMTokenRouter)
	Register(RouterCleanSILV12AgentL, NewCleanSILV12AgentLRouter)
	Register(RouterCleanSILV13AgentN, NewCleanSILV13AgentNRouter)
	Register(RouterCleanSILV13AgentP, NewCleanSILV13AgentPRouter)
	Register(RouterCleanSILV13AgentPTight, NewCleanSILV13AgentPTightRouter)
}

type evolvedPartitionTrackedRequest struct {
	podName      string
	tokens       float64
	decodeTokens float64
}

// evolvedPartitionRouter is the real-AIBrix adoption path for the SIL-evolved
// profile-shift heuristic. The current frozen variant is a soft token-aware
// global balancer: every tenant may use every replica, but small tenant and
// deployment penalties guide the router away from recurrent tail-latency traps.
type evolvedPartitionRouter struct {
	mu              sync.Mutex
	pendingTokens   map[string]float64
	pendingRequests map[string]int
}

func NewEvolvedPartitionRouter() (types.Router, error) {
	router := &evolvedPartitionRouter{
		pendingTokens:   make(map[string]float64),
		pendingRequests: make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("evolved-partition could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *evolvedPartitionRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	incomingTokens := r.promptLength(ctx)
	target, targetScore := r.bestPod(readyPods, tenant, incomingTokens)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, incomingTokens)
	klog.V(4).InfoS(
		"evolved_partition_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"incoming_tokens", incomingTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func tenantID(ctx *types.RoutingContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.ReqHeaders != nil {
		if tenant := strings.TrimSpace(ctx.ReqHeaders[evolvedPartitionTenantHeader]); tenant != "" {
			return tenant
		}
	}
	if ctx.User != nil {
		return *ctx.User
	}
	return ""
}

func traceRequestID(ctx *types.RoutingContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.ReqHeaders != nil {
		if requestID := strings.TrimSpace(ctx.ReqHeaders[evolvedPartitionTraceRequestHeader]); requestID != "" {
			return requestID
		}
	}
	return ctx.RequestID
}

func (r *evolvedPartitionRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *evolvedPartitionRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *evolvedPartitionRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *evolvedPartitionRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *evolvedPartitionRouter) bestPod(pods []*v1.Pod, tenant string, incomingTokens float64) (*v1.Pod, float64) {
	var best *v1.Pod
	bestScore := math.MaxFloat64
	for _, pod := range pods {
		score := r.score(pod, tenant, incomingTokens)
		if score < bestScore {
			best = pod
			bestScore = score
			continue
		}
		if math.Abs(score-bestScore) < 1e-9 && replicaID(pod) < replicaID(best) {
			best = pod
			bestScore = score
		}
	}
	return best, bestScore
}

func (r *evolvedPartitionRouter) score(pod *v1.Pod, tenant string, incomingTokens float64) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	podName := pod.Name
	id := replicaID(pod)
	return r.pendingTokens[podName] +
		350.0*float64(r.pendingRequests[podName]) +
		120.0*float64(r.pendingRequests[podName]) +
		0.20*incomingTokens +
		tenantPenalty(tenant, id) +
		hotReplicaPenalty(id)
}

func (r *evolvedPartitionRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, tokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  tokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[pod.Name] += tokens
	r.pendingRequests[pod.Name]++
}

func (r *evolvedPartitionRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[tracked.podName] -= tracked.tokens
	if r.pendingTokens[tracked.podName] < 0 {
		r.pendingTokens[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *evolvedPartitionRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	var body struct {
		Prompt   any `json:"prompt"`
		Messages []struct {
			Content any `json:"content"`
		} `json:"messages"`
	}
	if len(ctx.ReqBody) > 0 && json.Unmarshal(ctx.ReqBody, &body) == nil {
		if text := stringifyJSONText(body.Prompt); text != "" {
			return float64(estimateTokenishWords(text))
		}
		total := 0
		for _, message := range body.Messages {
			total += estimateTokenishWords(stringifyJSONText(message.Content))
		}
		if total > 0 {
			return float64(total)
		}
	}
	return 1.0
}

var trailingNumberRE = regexp.MustCompile(`(\d+)$`)

func tenantPenalty(tenant string, replica int) float64 {
	switch tenant {
	case "research":
		switch replica {
		case 1, 3:
			return 60.0
		}
	case "chat":
		switch replica {
		case 0:
			return 20.0
		case 2:
			return 40.0
		}
	case "code":
		switch replica {
		case 0, 2:
			return 60.0
		}
	}
	return 0.0
}

func hotReplicaPenalty(replica int) float64 {
	if replica == 2 {
		return 80.0
	}
	return 0.0
}

func replicaID(pod *v1.Pod) int {
	if pod == nil {
		return math.MaxInt32
	}
	for _, raw := range []string{
		pod.Labels["app.kubernetes.io/instance"],
		pod.Name,
	} {
		matches := trailingNumberRE.FindStringSubmatch(raw)
		if len(matches) == 2 {
			if value, err := strconv.Atoi(matches[1]); err == nil {
				return value
			}
		}
	}
	return math.MaxInt32
}

func stringifyJSONText(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			parts = append(parts, stringifyJSONText(item))
		}
		return strings.Join(parts, " ")
	case map[string]any:
		return stringifyJSONText(v["text"])
	default:
		return ""
	}
}

func estimateTokenishWords(text string) int {
	count := len(strings.Fields(text))
	if count <= 0 {
		return 1
	}
	return count
}

var _ types.Router = (*evolvedPartitionRouter)(nil)
var _ cache.RequestTracker = (*evolvedPartitionRouter)(nil)

// evolvedSILV2Router is the frozen SIL-evolved heuristic from
// scripts/case_policies/evolved_sil_router_v2.py. It is intentionally
// scenario-specialized for the 4-replica profile-shift demo and uses only
// gateway-maintainable request-tracker state.
type evolvedSILV2Router struct {
	mu              sync.Mutex
	pendingTokens   map[string]float64
	pendingRequests map[string]int
}

func NewEvolvedSILV2Router() (types.Router, error) {
	router := &evolvedSILV2Router{
		pendingTokens:   make(map[string]float64),
		pendingRequests: make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("evolved-sil-v2 could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *evolvedSILV2Router) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	incomingTokens := r.promptLength(ctx)
	preferredPods := make([]*v1.Pod, 0, len(readyPods))
	for _, pod := range readyPods {
		if r.softPenalty(tenant, replicaID(pod)) <= 1500.0 {
			preferredPods = append(preferredPods, pod)
		}
	}
	if len(preferredPods) == 0 {
		preferredPods = readyPods
	}

	bestPreferred, preferredScore := r.bestPod(preferredPods, tenant)
	bestGlobal, globalScore := r.bestPod(readyPods, tenant)
	if bestPreferred == nil || bestGlobal == nil {
		return "", ErrorNoAvailablePod
	}
	target := bestPreferred
	if target.Name != bestGlobal.Name &&
		r.loadScore(bestPreferred)-r.loadScore(bestGlobal) > 1500.0 {
		target = bestGlobal
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, incomingTokens)
	klog.V(4).InfoS(
		"evolved_sil_v2_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"incoming_tokens", incomingTokens,
		"preferred_score", preferredScore,
		"global_score", globalScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *evolvedSILV2Router) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *evolvedSILV2Router) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *evolvedSILV2Router) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *evolvedSILV2Router) SubscribedMetrics() []string {
	return []string{}
}

func (r *evolvedSILV2Router) bestPod(pods []*v1.Pod, tenant string) (*v1.Pod, float64) {
	var best *v1.Pod
	bestScore := math.MaxFloat64
	for _, pod := range pods {
		score := r.score(pod, tenant)
		if score < bestScore {
			best = pod
			bestScore = score
			continue
		}
		if math.Abs(score-bestScore) < 1e-9 && replicaID(pod) < replicaID(best) {
			best = pod
			bestScore = score
		}
	}
	return best, bestScore
}

func (r *evolvedSILV2Router) score(pod *v1.Pod, tenant string) float64 {
	return r.loadScore(pod) + r.softPenalty(tenant, replicaID(pod))
}

func (r *evolvedSILV2Router) loadScore(pod *v1.Pod) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	podName := pod.Name
	return r.pendingTokens[podName] + 190.0*float64(r.pendingRequests[podName])
}

func (r *evolvedSILV2Router) softPenalty(tenant string, replica int) float64 {
	switch tenant {
	case "research":
		switch replica {
		case 1, 3:
			return 2000.0
		}
	case "chat":
		switch replica {
		case 0, 2, 3:
			return 2000.0
		}
	case "code":
		switch replica {
		case 0, 2:
			return 2000.0
		}
	}
	return 0.0
}

func (r *evolvedSILV2Router) addPending(ctx *types.RoutingContext, pod *v1.Pod, tokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  tokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[pod.Name] += tokens
	r.pendingRequests[pod.Name]++
}

func (r *evolvedSILV2Router) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[tracked.podName] -= tracked.tokens
	if r.pendingTokens[tracked.podName] < 0 {
		r.pendingTokens[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *evolvedSILV2Router) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

var _ types.Router = (*evolvedSILV2Router)(nil)
var _ cache.RequestTracker = (*evolvedSILV2Router)(nil)

// evolvedOutstandingRouter is the most conservative SIL-evolved candidate:
// global least outstanding prompt-token load. It is less scenario-clever than
// evolved-sil-v2, but its state maps almost one-to-one onto a real gateway
// request tracker.
type evolvedOutstandingRouter struct {
	mu            sync.Mutex
	pendingTokens map[string]float64
}

func NewEvolvedOutstandingRouter() (types.Router, error) {
	router := &evolvedOutstandingRouter{
		pendingTokens: make(map[string]float64),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("evolved-outstanding could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *evolvedOutstandingRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}
	incomingTokens := r.promptLength(ctx)
	target, targetScore := r.bestPod(readyPods)
	if target == nil {
		return "", ErrorNoAvailablePod
	}
	ctx.SetTargetPod(target)
	r.addPending(ctx, target, incomingTokens)
	klog.V(4).InfoS(
		"evolved_outstanding_route",
		"request_id", ctx.RequestID,
		"target_pod", target.Name,
		"incoming_tokens", incomingTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *evolvedOutstandingRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *evolvedOutstandingRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *evolvedOutstandingRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *evolvedOutstandingRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *evolvedOutstandingRouter) bestPod(pods []*v1.Pod) (*v1.Pod, float64) {
	var best *v1.Pod
	bestScore := math.MaxFloat64
	for _, pod := range pods {
		score := r.loadScore(pod)
		if score < bestScore {
			best = pod
			bestScore = score
			continue
		}
		if math.Abs(score-bestScore) < 1e-9 && replicaID(pod) < replicaID(best) {
			best = pod
			bestScore = score
		}
	}
	return best, bestScore
}

func (r *evolvedOutstandingRouter) loadScore(pod *v1.Pod) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pendingTokens[pod.Name]
}

func (r *evolvedOutstandingRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, tokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  tokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[pod.Name] += tokens
}

func (r *evolvedOutstandingRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[tracked.podName] -= tracked.tokens
	if r.pendingTokens[tracked.podName] < 0 {
		r.pendingTokens[tracked.podName] = 0
	}
}

func (r *evolvedOutstandingRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

var _ types.Router = (*evolvedOutstandingRouter)(nil)
var _ cache.RequestTracker = (*evolvedOutstandingRouter)(nil)

// e2ePortableAgentRouter is the frozen SIL-evolved routing-only E2E candidate
// for the long-window profile-shift demo. It is metric-guarded and
// Go-portable: AIBrix's cached GPUBusyTimeRatio/num_requests_running metric is
// the primary score, while gateway-maintained outstanding prompt tokens are
// only used as a token-aware tie/near-tie smoother.
type e2ePortableAgentRouter struct {
	mu              sync.Mutex
	cache           cache.Cache
	pendingTokens   map[string]float64
	pendingRequests map[string]int
}

func NewE2EPortableAgentRouter() (types.Router, error) {
	c, err := cache.Get()
	router := &e2ePortableAgentRouter{
		cache:           c,
		pendingTokens:   make(map[string]float64),
		pendingRequests: make(map[string]int),
	}
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("e2e-portable-agent could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *e2ePortableAgentRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	incomingTokens := r.promptLength(ctx)
	target, targetScore := r.bestPod(readyPods)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, incomingTokens)
	klog.V(4).InfoS(
		"e2e_portable_agent_route",
		"request_id", ctx.RequestID,
		"tenant", tenantID(ctx),
		"target_pod", target.Name,
		"incoming_tokens", incomingTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *e2ePortableAgentRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *e2ePortableAgentRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *e2ePortableAgentRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *e2ePortableAgentRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *e2ePortableAgentRouter) bestPod(pods []*v1.Pod) (*v1.Pod, float64) {
	bestScore := math.MaxFloat64
	var best *v1.Pod
	for _, pod := range pods {
		score := r.loadScore(pod)
		if score < bestScore {
			bestScore = score
			best = pod
			continue
		}
		if math.Abs(score-bestScore) < 1e-9 && replicaID(pod) < replicaID(best) {
			best = pod
		}
	}
	if best == nil {
		return nil, bestScore
	}
	return best, bestScore
}

func (r *e2ePortableAgentRouter) loadScore(pod *v1.Pod) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return 10000.0*r.busyScore(pod) + r.pendingTokens[pod.Name]
}

func (r *e2ePortableAgentRouter) busyScore(pod *v1.Pod) float64 {
	if r.cache == nil || pod == nil {
		return 0.0
	}
	value, err := r.cache.GetMetricValueByPod(pod.Name, pod.Namespace, metrics.GPUBusyTimeRatio)
	if err != nil || value == nil {
		return 0.0
	}
	score := value.GetSimpleValue()
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0.0
	}
	return score
}

func (r *e2ePortableAgentRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, tokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  tokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[pod.Name] += tokens
	r.pendingRequests[pod.Name]++
}

func (r *e2ePortableAgentRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[tracked.podName] -= tracked.tokens
	if r.pendingTokens[tracked.podName] < 0 {
		r.pendingTokens[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *e2ePortableAgentRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

var _ types.Router = (*e2ePortableAgentRouter)(nil)
var _ cache.RequestTracker = (*e2ePortableAgentRouter)(nil)

// e2eGPUPartitionRouter is the 8-replica routing-only E2E candidate evolved in
// SIL for the small-model / two-replicas-per-A6000 setup.  It deliberately
// reserves whole GPU lanes for the tight-SLO chat tenant while keeping code and
// research on looser-SLO lanes.  The router uses only gateway-portable signals:
// tenant header, trace request id, endpoint GPU labels, and an in-gateway
// outstanding request/token tracker.
type e2eGPUPartitionRouter struct {
	mu              sync.Mutex
	pendingTokens   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewE2EGPUPartitionRouter() (types.Router, error) {
	router := &e2eGPUPartitionRouter{
		pendingTokens:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("e2e-gpu-partition could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *e2eGPUPartitionRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	incomingTokens := r.promptLength(ctx)
	preferredPods := make([]*v1.Pod, 0, len(readyPods))
	for _, pod := range readyPods {
		if r.prefersGPU(tenant, incomingTokens, podGPUID(pod)) {
			preferredPods = append(preferredPods, pod)
		}
	}
	if len(preferredPods) == 0 {
		preferredPods = readyPods
	}

	target, targetScore := r.bestPod(ctx, preferredPods, tenant)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, incomingTokens)
	klog.V(4).InfoS(
		"e2e_gpu_partition_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"incoming_tokens", incomingTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *e2eGPUPartitionRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *e2eGPUPartitionRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *e2eGPUPartitionRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *e2eGPUPartitionRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *e2eGPUPartitionRouter) prefersGPU(tenant string, promptTokens float64, gpuID int) bool {
	if promptTokens >= 520.0 {
		return gpuID == 2 || gpuID == 3
	}
	switch tenant {
	case "chat":
		return gpuID == 0 || gpuID == 1 || gpuID == 2
	case "code":
		return gpuID == 2 || gpuID == 3
	case "research":
		return gpuID == 3
	default:
		return true
	}
}

func (r *e2eGPUPartitionRouter) bestPod(ctx *types.RoutingContext, pods []*v1.Pod, tenant string) (*v1.Pod, float64) {
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant)
		if score < bestScore-0.001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if math.Abs(score-bestScore) <= 0.001 {
			tied = append(tied, pod)
		}
	}
	if len(tied) == 0 {
		return nil, bestScore
	}
	for i := 0; i < len(tied); i++ {
		for j := i + 1; j < len(tied); j++ {
			if replicaID(tied[j]) < replicaID(tied[i]) {
				tied[i], tied[j] = tied[j], tied[i]
			}
		}
	}
	return tied[e2eLCGTieIndex(traceRequestID(ctx), len(tied), 971)], bestScore
}

func (r *e2eGPUPartitionRouter) score(pod *v1.Pod, tenant string) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID
	requestWeight := 70.0
	sameScale := 1.0
	switch tenant {
	case "chat":
		requestWeight = 115.0
		sameScale = 1.4
	case "code":
		requestWeight = 26.0
	case "research":
		requestWeight = 34.0
	}

	ownTokens := r.pendingTokens[podName]
	ownRequests := float64(r.pendingRequests[podName])
	siblingTokens := 0.0
	siblingRequests := 0.0
	for otherPodName, tokens := range r.pendingTokens {
		if otherPodName == podName {
			continue
		}
		if r.podGPU[otherPodName] == gpuID {
			siblingTokens += tokens
			siblingRequests += float64(r.pendingRequests[otherPodName])
		}
	}

	return requestWeight*ownRequests +
		ownTokens +
		sameScale*24.0*siblingRequests +
		sameScale*0.35*siblingTokens
}

func (r *e2eGPUPartitionRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, tokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  tokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingTokens[pod.Name] += tokens
	r.pendingRequests[pod.Name]++
}

func (r *e2eGPUPartitionRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingTokens[tracked.podName] -= tracked.tokens
	if r.pendingTokens[tracked.podName] < 0 {
		r.pendingTokens[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *e2eGPUPartitionRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	var body struct {
		Prompt   any `json:"prompt"`
		Messages []struct {
			Content any `json:"content"`
		} `json:"messages"`
	}
	if len(ctx.ReqBody) > 0 && json.Unmarshal(ctx.ReqBody, &body) == nil {
		if text := stringifyJSONText(body.Prompt); text != "" {
			return float64(estimateTokenishWords(text))
		}
		total := 0
		for _, message := range body.Messages {
			total += estimateTokenishWords(stringifyJSONText(message.Content))
		}
		if total > 0 {
			return float64(total)
		}
	}
	return 1.0
}

var _ types.Router = (*e2eGPUPartitionRouter)(nil)
var _ cache.RequestTracker = (*e2eGPUPartitionRouter)(nil)

// silV6CV2Router is the real-AIBrix port of
// scripts/case_policies/sil_loop_v6c_multi_metric_router_v2.py.  It is
// intentionally scenario-specialized for the frozen v6c 8-replica trace, and
// only uses portable gateway inputs: tenant/request headers, endpoint GPU
// labels, request prompt/max-token hints, and an in-gateway outstanding load
// tracker.
type silV6CV2Router struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewSILV6CV2Router() (types.Router, error) {
	router := &silV6CV2Router{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("sil-v6c-v2 could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *silV6CV2Router) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	preferredPods := make([]*v1.Pod, 0, len(readyPods))
	for _, pod := range readyPods {
		if r.prefersGPU(tenant, promptTokens, podGPUID(pod)) {
			preferredPods = append(preferredPods, pod)
		}
	}
	if len(preferredPods) == 0 {
		preferredPods = readyPods
	}

	bestPreferred, preferredScore := r.bestPod(ctx, preferredPods, tenant)
	bestGlobal, globalScore := r.bestPod(ctx, readyPods, tenant)
	if bestPreferred == nil || bestGlobal == nil {
		return "", ErrorNoAvailablePod
	}
	target := bestPreferred
	if target.Name != bestGlobal.Name && r.shouldOverflow(preferredScore, globalScore) {
		target = bestGlobal
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"sil_v6c_v2_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"preferred_score", preferredScore,
		"global_score", globalScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *silV6CV2Router) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *silV6CV2Router) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *silV6CV2Router) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *silV6CV2Router) SubscribedMetrics() []string {
	return []string{}
}

func (r *silV6CV2Router) prefersGPU(tenant string, promptTokens float64, gpuID int) bool {
	if promptTokens >= 520.0 && (gpuID == 2 || gpuID == 3) {
		return true
	}
	switch tenant {
	case "chat":
		return gpuID == 0 || gpuID == 1
	case "code":
		return gpuID == 2 || gpuID == 3
	case "research":
		return gpuID == 0 || gpuID == 2 || gpuID == 3
	default:
		return true
	}
}

func (r *silV6CV2Router) bestPod(ctx *types.RoutingContext, pods []*v1.Pod, tenant string) (*v1.Pod, float64) {
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant)
		if score < bestScore-0.001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if math.Abs(score-bestScore) <= 0.001 {
			tied = append(tied, pod)
		}
	}
	if len(tied) == 0 {
		return nil, bestScore
	}
	for i := 0; i < len(tied); i++ {
		for j := i + 1; j < len(tied); j++ {
			if replicaID(tied[j]) < replicaID(tied[i]) {
				tied[i], tied[j] = tied[j], tied[i]
			}
		}
	}
	return tied[silV6CV2TieIndex(traceRequestID(ctx), len(tied), 999)], bestScore
}

func (r *silV6CV2Router) score(pod *v1.Pod, tenant string) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID

	requestWeight := 72.0
	sameScale := 1.0
	switch tenant {
	case "chat":
		requestWeight = 135.0
		sameScale = 1.65
	case "code":
		requestWeight = 30.0
	case "research":
		requestWeight = 32.0
	}

	ownRequests := float64(r.pendingRequests[podName])
	ownPrefill := r.pendingPrefill[podName]
	ownDecode := r.pendingDecode[podName]
	siblingRequests := 0.0
	siblingPrefill := 0.0
	siblingDecode := 0.0
	for otherPodName, prefillTokens := range r.pendingPrefill {
		if otherPodName == podName {
			continue
		}
		if r.podGPU[otherPodName] == gpuID {
			siblingRequests += float64(r.pendingRequests[otherPodName])
			siblingPrefill += prefillTokens
			siblingDecode += r.pendingDecode[otherPodName]
		}
	}

	return requestWeight*ownRequests +
		ownPrefill +
		1.2*ownDecode +
		sameScale*28.0*siblingRequests +
		sameScale*0.35*siblingPrefill +
		sameScale*0.35*siblingDecode
}

func (r *silV6CV2Router) shouldOverflow(preferredScore, globalScore float64) bool {
	if preferredScore-globalScore <= 1000000000.0 {
		return false
	}
	if globalScore <= 0.0 {
		return true
	}
	return preferredScore/globalScore >= 1000000.0
}

func (r *silV6CV2Router) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *silV6CV2Router) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *silV6CV2Router) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *silV6CV2Router) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

var _ types.Router = (*silV6CV2Router)(nil)
var _ cache.RequestTracker = (*silV6CV2Router)(nil)

// cleanSILV3AgentBRouter is the frozen Go port of
// scripts/case_policies/clean_sil_v3_agent_b_router.py. It is intentionally
// scenario-specialized for the clean SIL v3 / Qwen-1.5B / 8-replica replay and
// only uses portable gateway signals: tenant/request headers, endpoint GPU
// identity, and in-gateway outstanding request/token counters.
type cleanSILV3AgentBRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV3AgentBRouter() (types.Router, error) {
	router := &cleanSILV3AgentBRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v3-agent-b could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV3AgentBRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)

	preferredPods := r.preferredPods(readyPods, tenant, promptTokens)
	bestPreferred, preferredScore := r.bestPod(ctx, preferredPods, tenant)
	bestGlobal, globalScore := r.bestPod(ctx, readyPods, tenant)
	if bestPreferred == nil || bestGlobal == nil {
		return "", ErrorNoAvailablePod
	}
	target := bestPreferred
	if target.Name != bestGlobal.Name && r.shouldOverflow(preferredScore, globalScore) {
		target = bestGlobal
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v3_agent_b_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"preferred_score", preferredScore,
		"global_score", globalScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV3AgentBRouter) preferredPods(pods []*v1.Pod, tenant string, promptTokens float64) []*v1.Pod {
	preferredGPUs := map[int]bool{}
	for _, gpuID := range []int{0, 1, 2, 3} {
		preferredGPUs[gpuID] = true
	}
	if promptTokens >= 360.0 {
		for _, gpuID := range []int{0, 1, 2, 3} {
			preferredGPUs[gpuID] = true
		}
	}
	out := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		if preferredGPUs[podGPUID(pod)] {
			out = append(out, pod)
		}
	}
	if len(out) == 0 {
		return pods
	}
	return out
}

func (r *cleanSILV3AgentBRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV3AgentBRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV3AgentBRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV3AgentBRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV3AgentBRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV3AgentBRouter) bestPod(ctx *types.RoutingContext, pods []*v1.Pod, tenant string) (*v1.Pod, float64) {
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant)
		if score < bestScore-0.001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if score-bestScore <= 0.001 {
			tied = append(tied, pod)
		}
	}
	if len(tied) == 0 {
		return nil, bestScore
	}
	for i := 0; i < len(tied); i++ {
		for j := i + 1; j < len(tied); j++ {
			if replicaID(tied[j]) < replicaID(tied[i]) {
				tied[i], tied[j] = tied[j], tied[i]
			}
		}
	}
	return tied[silV6CV2TieIndex(traceRequestID(ctx), len(tied), 999)], bestScore
}

func (r *cleanSILV3AgentBRouter) score(pod *v1.Pod, tenant string) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID

	requestWeight := 60.0
	sameScale := 1.0
	switch tenant {
	case "chat":
		requestWeight = 90.0
		sameScale = 1.2
	case "code":
		requestWeight = 55.0
	case "research":
		requestWeight = 50.0
	}

	ownRequests := float64(r.pendingRequests[podName])
	ownPrefill := r.pendingPrefill[podName]
	ownDecode := r.pendingDecode[podName]
	siblingRequests := 0.0
	siblingPrefill := 0.0
	siblingDecode := 0.0
	for otherPodName, prefillTokens := range r.pendingPrefill {
		if otherPodName == podName {
			continue
		}
		if r.podGPU[otherPodName] == gpuID {
			siblingRequests += float64(r.pendingRequests[otherPodName])
			siblingPrefill += prefillTokens
			siblingDecode += r.pendingDecode[otherPodName]
		}
	}

	score := requestWeight*ownRequests +
		ownPrefill +
		ownDecode +
		sameScale*20.0*siblingRequests +
		sameScale*0.25*siblingPrefill +
		sameScale*0.25*siblingDecode

	if tenant == "research" {
		score += excessFloat(siblingDecode, 90.0) * 0.35
		score += excessFloat(siblingRequests, 1.0) * 18.0
	}
	return score
}

func (r *cleanSILV3AgentBRouter) shouldOverflow(preferredScore, globalScore float64) bool {
	if preferredScore-globalScore <= 1000000000.0 {
		return false
	}
	if globalScore <= 0.0 {
		return true
	}
	return preferredScore/globalScore >= 1000000.0
}

func (r *cleanSILV3AgentBRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV3AgentBRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV3AgentBRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV3AgentBRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

var _ types.Router = (*cleanSILV3AgentBRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV3AgentBRouter)(nil)

// cleanSILV8AgentKRouter is the frozen Go port of
// scripts/case_policies/clean_sil_v8_agent_k_router.py with the SIL-selected
// candidate_009 parameters for the v6c/s008/clipped430 8-replica E2E gate.
// It stays anchored to least-request, then lets code/research escape to a
// lower same-GPU pressure replica when the gateway tracker says that is safer.
type cleanSILV8AgentKRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV8AgentKRouter() (types.Router, error) {
	router := &cleanSILV8AgentKRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v8-agent-k could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV8AgentKRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)

	minRequests := math.MaxFloat64
	for _, pod := range readyPods {
		requests := r.requests(pod)
		if requests < minRequests {
			minRequests = requests
		}
	}
	anchor := r.leastRequestAnchor(readyPods, minRequests, tenant, requestID)
	if anchor == nil {
		return "", ErrorNoAvailablePod
	}
	if tenant == "chat" || (tenant != "code" && tenant != "research") {
		ctx.SetTargetPod(anchor)
		r.addPending(ctx, anchor, promptTokens, decodeTokens)
		return ctx.TargetAddress(), nil
	}

	allowed := make([]*v1.Pod, 0, len(readyPods))
	for _, pod := range readyPods {
		if r.requests(pod) <= minRequests+1.0 {
			allowed = append(allowed, pod)
		}
	}
	best := r.bestPod(allowed, tenant, promptTokens, decodeTokens, requestID)
	target := anchor
	if best != nil && best.Name != anchor.Name && r.shouldEscape(anchor, best, tenant, promptTokens, decodeTokens) {
		target = best
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v8_agent_k_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV8AgentKRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV8AgentKRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV8AgentKRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV8AgentKRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV8AgentKRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV8AgentKRouter) leastRequestAnchor(pods []*v1.Pod, minRequests float64, tenant string, requestID string) *v1.Pod {
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		if math.Abs(r.requests(pod)-minRequests) < 1e-9 {
			tied = append(tied, pod)
		}
	}
	sortPodsByReplicaID(tied)
	if len(tied) == 0 {
		return nil
	}
	return tied[cleanSILV8TieIndex(requestID, tenant, len(tied), 1207)]
}

func (r *cleanSILV8AgentKRouter) bestPod(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64, requestID string) *v1.Pod {
	if len(pods) == 0 {
		return nil
	}
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant, promptTokens, decodeTokens)
		if score < bestScore-0.001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if score-bestScore <= 0.001 {
			tied = append(tied, pod)
		}
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV8TieIndex(requestID, tenant, len(tied), 1207)]
}

func (r *cleanSILV8AgentKRouter) score(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID

	requestScale, prefillScale, decodeScale, sameGPUScale := 1.0, 1.0, 1.0, 1.0
	switch tenant {
	case "chat":
		requestScale, prefillScale, decodeScale, sameGPUScale = 1.25, 0.70, 0.70, 1.15
	case "code":
		requestScale, prefillScale, decodeScale, sameGPUScale = 1.02, 1.00, 0.98, 0.92
	case "research":
		requestScale, prefillScale, decodeScale, sameGPUScale = 0.90, 1.05, 1.24, 1.32
		if decodeTokens >= 70.0 {
			sameGPUScale += 0.10
			decodeScale += 0.10
		}
	}

	ownRequests := float64(r.pendingRequests[podName])
	ownPrefill := r.pendingPrefill[podName]
	ownDecode := r.pendingDecode[podName]
	siblingRequests, siblingPrefill, siblingDecode := r.sameGPUOutstandingLocked(podName, gpuID)

	return requestScale*100.0*ownRequests +
		prefillScale*0.22*ownPrefill +
		decodeScale*0.50*ownDecode +
		sameGPUScale*24.0*siblingRequests +
		sameGPUScale*0.045*siblingPrefill +
		sameGPUScale*0.22*siblingDecode +
		cleanSILV8TenantGPUPenalty(tenant, gpuID)
}

func (r *cleanSILV8AgentKRouter) shouldEscape(anchor, best *v1.Pod, tenant string, promptTokens, decodeTokens float64) bool {
	requestGap := r.requests(best) - r.requests(anchor)
	anchorScore := r.score(anchor, tenant, promptTokens, decodeTokens)
	bestScore := r.score(best, tenant, promptTokens, decodeTokens)
	if anchorScore-bestScore < 0.0 {
		return false
	}
	if anchorScore > 0.0 && bestScore/anchorScore > 1.0 {
		return false
	}

	anchorGPU := podGPUID(anchor)
	bestGPU := podGPUID(best)
	r.mu.Lock()
	anchorSameRequests, anchorSamePrefill, anchorSameDecode := r.sameGPUOutstandingLocked(anchor.Name, anchorGPU)
	bestSameRequests, bestSamePrefill, bestSameDecode := r.sameGPUOutstandingLocked(best.Name, bestGPU)
	r.mu.Unlock()

	requestDrop := anchorSameRequests - bestSameRequests
	prefillDrop := anchorSamePrefill - bestSamePrefill
	decodeDrop := anchorSameDecode - bestSameDecode
	if requestGap <= 1e-9 {
		return requestDrop >= 0.0 || prefillDrop >= 0.0 || decodeDrop >= 0.0
	}
	return requestDrop >= 0.0 || prefillDrop >= 0.0 || decodeDrop >= 0.0
}

func (r *cleanSILV8AgentKRouter) requests(pod *v1.Pod) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return float64(r.pendingRequests[pod.Name])
}

func (r *cleanSILV8AgentKRouter) sameGPUOutstandingLocked(podName string, gpuID int) (float64, float64, float64) {
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV8AgentKRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  promptTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV8AgentKRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	decodeTokens := r.outputLength(ctx)
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV8AgentKRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV8AgentKRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

func cleanSILV8TenantGPUPenalty(tenant string, gpuID int) float64 {
	switch tenant {
	case "research":
		if gpuID == 2 || gpuID == 3 {
			return -10.0
		}
	case "code":
		if gpuID == 0 || gpuID == 1 {
			return -6.0
		}
	}
	return 0.0
}

func sortPodsByReplicaID(pods []*v1.Pod) {
	for i := 0; i < len(pods); i++ {
		for j := i + 1; j < len(pods); j++ {
			if replicaID(pods[j]) < replicaID(pods[i]) {
				pods[i], pods[j] = pods[j], pods[i]
			}
		}
	}
}

func cleanSILV8TieIndex(requestID string, tenant string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	tenantOffset := 0
	switch tenant {
	case "chat":
		tenantOffset = 17
	case "code":
		tenantOffset = 43
	case "research":
		tenantOffset = 71
	}
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		return (((value + 1) * 1103515245) + 12345 + salt + tenantOffset) % count
	}
	return stableReplicaIndex(requestID+":"+tenant+":"+strconv.Itoa(salt), count)
}

var _ types.Router = (*cleanSILV8AgentKRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV8AgentKRouter)(nil)

// cleanSILV8AgentSRouter is the frozen Go port of
// scripts/case_policies/clean_sil_v8_agent_s_router.py for the clean
// v6c/s006/clipped430/full7600 SIL-only evolution gate.
type cleanSILV8AgentSRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV8AgentSRouter() (types.Router, error) {
	router := &cleanSILV8AgentSRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v8-agent-s could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV8AgentSRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)

	target := r.bestPod(readyPods, tenant, promptTokens, decodeTokens, requestID)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v8_agent_s_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV8AgentSRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV8AgentSRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV8AgentSRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV8AgentSRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV8AgentSRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV8AgentSRouter) bestPod(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64, requestID string) *v1.Pod {
	if len(pods) == 0 {
		return nil
	}
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant, promptTokens, decodeTokens)
		if score < bestScore-0.001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if score-bestScore <= 0.001 {
			tied = append(tied, pod)
		}
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV8TieIndex(requestID, tenant, len(tied), 3067)]
}

func (r *cleanSILV8AgentSRouter) score(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID

	ownRequests := float64(r.pendingRequests[podName])
	ownPrefill := r.pendingPrefill[podName]
	ownDecode := r.pendingDecode[podName]
	siblingRequests, siblingPrefill, siblingDecode := r.sameGPUOutstandingLocked(podName, gpuID)
	sameGPUScale := cleanSILV8AgentSSameGPUScale(tenant)
	penalty := cleanSILV8AgentSGPUPenalty(tenant, gpuID)
	if promptTokens >= 300.0 {
		penalty += cleanSILV8AgentSLongPromptBonus(gpuID)
	}

	return penalty +
		30.0*ownRequests +
		0.45*ownPrefill +
		2.95*ownDecode +
		sameGPUScale*58.0*siblingRequests +
		sameGPUScale*0.20*siblingPrefill +
		sameGPUScale*0.90*siblingDecode +
		0.06*promptTokens +
		0.24*decodeTokens
}

func cleanSILV8AgentSGPUPenalty(tenant string, gpuID int) float64 {
	switch tenant {
	case "chat":
		switch gpuID {
		case 0:
			return -2.0
		case 1:
			return 15.0
		case 2:
			return 220.0
		case 3:
			return 250.0
		}
	case "code":
		switch gpuID {
		case 0:
			return 80.0
		case 1:
			return -5.0
		case 2:
			return 180.0
		case 3:
			return -25.0
		}
	case "research":
		switch gpuID {
		case 0:
			return 0.0
		case 1:
			return -25.0
		case 2:
			return 320.0
		case 3:
			return -65.0
		}
	}
	return 150.0
}

func cleanSILV8AgentSSameGPUScale(tenant string) float64 {
	switch tenant {
	case "chat":
		return 1.70
	case "code":
		return 0.92
	case "research":
		return 2.25
	}
	return 1.0
}

func cleanSILV8AgentSLongPromptBonus(gpuID int) float64 {
	switch gpuID {
	case 2:
		return 150.0
	case 3:
		return -95.0
	}
	return 0.0
}

func (r *cleanSILV8AgentSRouter) sameGPUOutstandingLocked(podName string, gpuID int) (float64, float64, float64) {
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV8AgentSRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  promptTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV8AgentSRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	decodeTokens := r.outputLength(ctx)
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV8AgentSRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV8AgentSRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

var _ types.Router = (*cleanSILV8AgentSRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV8AgentSRouter)(nil)

// cleanSILV9AgentVRouter is the frozen Go port of
// scripts/case_policies/clean_sil_v9_agent_v_router.py with the selected
// power_score parameters from the clean token-burst SIL-only evolution gate.
// It uses only gateway-portable request tracker state and endpoint GPU labels.
type cleanSILV9AgentVRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV9AgentVRouter() (types.Router, error) {
	router := &cleanSILV9AgentVRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v9-agent-v could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV9AgentVRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)

	target := r.bestPod(readyPods, tenant, promptTokens, decodeTokens, requestID)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v9_agent_v_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV9AgentVRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV9AgentVRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV9AgentVRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV9AgentVRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV9AgentVRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV9AgentVRouter) bestPod(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64, requestID string) *v1.Pod {
	if len(pods) == 0 {
		return nil
	}
	k := cleanSILV9AgentVPowerK(tenant)
	if k > len(pods) {
		k = len(pods)
	}
	sampled := append([]*v1.Pod(nil), pods...)
	for i := 0; i < len(sampled); i++ {
		for j := i + 1; j < len(sampled); j++ {
			hi := cleanSILV9AgentVHash(requestID, tenant, replicaID(sampled[i]), 20290529)
			hj := cleanSILV9AgentVHash(requestID, tenant, replicaID(sampled[j]), 20290529)
			if hj < hi || (hj == hi && replicaID(sampled[j]) < replicaID(sampled[i])) {
				sampled[i], sampled[j] = sampled[j], sampled[i]
			}
		}
	}
	return r.bestFrom(sampled[:k], tenant, promptTokens, decodeTokens, requestID)
}

func (r *cleanSILV9AgentVRouter) bestFrom(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64, requestID string) *v1.Pod {
	if len(pods) == 0 {
		return nil
	}
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant, promptTokens, decodeTokens, requestID)
		if score < bestScore-0.001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if score-bestScore <= 0.001 {
			tied = append(tied, pod)
		}
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV9AgentVTieIndex(requestID, tenant, len(tied), 20290529)]
}

func (r *cleanSILV9AgentVRouter) score(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64, requestID string) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID

	requestScale, prefillScale, decodeScale, sameGPUScale := cleanSILV9AgentVTenantScales(tenant)
	if tenant == "code" && promptTokens >= 320.0 {
		prefillScale += 0.08
		sameGPUScale += 0.05
	}
	if tenant == "research" && decodeTokens >= 62.0 {
		decodeScale += 0.10
		sameGPUScale += 0.10
	}

	ownRequests := float64(r.pendingRequests[podName])
	ownPrefill := r.pendingPrefill[podName]
	ownDecode := r.pendingDecode[podName]
	siblingRequests, siblingPrefill, siblingDecode := r.sameGPUOutstandingLocked(podName, gpuID)

	incomingTokens := 0.055*promptTokens + 0.16*decodeTokens
	incomingPressure := incomingTokens * (ownRequests +
		0.32*siblingRequests +
		0.00050*siblingPrefill +
		0.0045*siblingDecode)

	score := cleanSILV9AgentVGPUBias(tenant, gpuID, promptTokens, decodeTokens) +
		requestScale*45.0*ownRequests +
		prefillScale*0.32*ownPrefill +
		decodeScale*0.60*ownDecode +
		sameGPUScale*14.0*siblingRequests +
		sameGPUScale*0.04*siblingPrefill +
		sameGPUScale*0.14*siblingDecode +
		incomingPressure

	if siblingRequests >= 3.0 {
		score += 10.0
	}
	if siblingDecode >= 135.0 {
		score += 8.0
	}
	return score
}

func cleanSILV9AgentVPowerK(tenant string) int {
	switch tenant {
	case "chat", "code":
		return 2
	case "research":
		return 3
	}
	return 3
}

func cleanSILV9AgentVTenantScales(tenant string) (float64, float64, float64, float64) {
	switch tenant {
	case "chat":
		return 1.20, 0.58, 0.72, 1.18
	case "code":
		return 1.00, 1.12, 1.02, 0.98
	case "research":
		return 0.94, 0.96, 1.22, 1.24
	}
	return 1.0, 1.0, 1.0, 1.0
}

func cleanSILV9AgentVGPUBias(tenant string, gpuID int, promptTokens, decodeTokens float64) float64 {
	bias := 0.0
	switch tenant {
	case "chat":
		switch gpuID {
		case 0:
			bias = -8.0
		case 1:
			bias = -4.0
		case 2:
			bias = 20.0
		case 3:
			bias = 16.0
		}
	case "code":
		switch gpuID {
		case 0:
			bias = 14.0
		case 1:
			bias = -8.0
		case 2:
			bias = 10.0
		case 3:
			bias = -10.0
		}
		if promptTokens >= 320.0 {
			switch gpuID {
			case 0:
				bias += 10.0
			case 1:
				bias += -4.0
			case 2:
				bias += 8.0
			case 3:
				bias += -10.0
			}
		}
	case "research":
		switch gpuID {
		case 0:
			bias = -4.0
		case 1:
			bias = 8.0
		case 2:
			bias = 24.0
		case 3:
			bias = -8.0
		}
		if decodeTokens >= 62.0 {
			switch gpuID {
			case 0:
				bias += -4.0
			case 1:
				bias += 4.0
			case 2:
				bias += 18.0
			case 3:
				bias += -10.0
			}
		}
	}
	return bias
}

func (r *cleanSILV9AgentVRouter) sameGPUOutstandingLocked(podName string, gpuID int) (float64, float64, float64) {
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV9AgentVRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  promptTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV9AgentVRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	decodeTokens := r.outputLength(ctx)
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV9AgentVRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV9AgentVRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

func cleanSILV9AgentVTieIndex(requestID string, tenant string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	return int(cleanSILV9AgentVHash(requestID, tenant, salt, salt) % int64(count))
}

func cleanSILV9AgentVHash(requestID string, tenant string, value int, salt int) int64 {
	requestValue := int64(0)
	if parsed, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		requestValue = int64(parsed)
	} else {
		return int64(stableReplicaIndex(requestID+":"+tenant+":"+strconv.Itoa(value)+":"+strconv.Itoa(salt), math.MaxInt32))
	}
	tenantOffset := int64(0)
	switch tenant {
	case "chat":
		tenantOffset = 17
	case "code":
		tenantOffset = 43
	case "research":
		tenantOffset = 71
	}
	x := (requestValue+1)*1103515245 + 12345
	x += int64(value)*2654435761 + tenantOffset + int64(salt)
	return x & 0x7FFFFFFF
}

var _ types.Router = (*cleanSILV9AgentVRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV9AgentVRouter)(nil)

// cleanSILV10AgentYRouter is the Go port of
// scripts/case_policies/clean_sil_v10_agent_y_router.py. It uses gateway
// outstanding counters plus endpoint placement labels, and preserves the
// Python policy's soft tenant lane + overflow scoring surface.
type cleanSILV10AgentYRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV10AgentYRouter() (types.Router, error) {
	router := &cleanSILV10AgentYRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v10-agent-y could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV10AgentYRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)

	preferred := r.preferredPods(readyPods, tenant)
	bestLane := r.bestPod(r.samplePods(preferred, tenant, requestID), tenant, promptTokens, decodeTokens, requestID)
	bestGlobal := r.bestPod(r.samplePods(readyPods, tenant, requestID), tenant, promptTokens, decodeTokens, requestID)
	if bestLane == nil {
		bestLane = bestGlobal
	}
	target := bestLane
	if bestGlobal != nil && bestLane != nil && bestLane.Name != bestGlobal.Name {
		laneScore := r.score(bestLane, tenant, promptTokens, decodeTokens)
		globalScore := r.score(bestGlobal, tenant, promptTokens, decodeTokens)
		if cleanSILV10ShouldOverflow(laneScore, globalScore) {
			target = bestGlobal
		}
	}
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v10_agent_y_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV10AgentYRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV10AgentYRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV10AgentYRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV10AgentYRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV10AgentYRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV10AgentYRouter) preferredPods(pods []*v1.Pod, tenant string) []*v1.Pod {
	preferred := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		replica := replicaID(pod)
		switch tenant {
		case "code":
			if replica == 1 || replica == 3 || replica == 5 || replica == 7 {
				preferred = append(preferred, pod)
			}
		case "research":
			if replica == 0 || replica == 1 || replica == 6 || replica == 7 {
				preferred = append(preferred, pod)
			}
		default:
			preferred = append(preferred, pod)
		}
	}
	if len(preferred) == 0 {
		return pods
	}
	return preferred
}

func (r *cleanSILV10AgentYRouter) bestPod(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64, requestID string) *v1.Pod {
	if len(pods) == 0 {
		return nil
	}
	bestScore := math.MaxFloat64
	tied := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		score := r.score(pod, tenant, promptTokens, decodeTokens)
		if score < bestScore-0.0001 {
			bestScore = score
			tied = tied[:0]
			tied = append(tied, pod)
			continue
		}
		if score-bestScore <= 0.0001 {
			tied = append(tied, pod)
		}
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV10TieIndex(requestID, tenant, len(tied), 10091)]
}

func (r *cleanSILV10AgentYRouter) score(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	podName := pod.Name
	gpuID := podGPUID(pod)
	r.podGPU[podName] = gpuID
	requests := float64(r.pendingRequests[podName])
	prefill := r.pendingPrefill[podName]
	decode := r.pendingDecode[podName]
	sameRequests, samePrefill, sameDecode := r.sameGPUOutstandingLocked(podName, gpuID)

	reqScale, prefillScale, decodeScale, sameScale, slackScale := cleanSILV10Scales(tenant, promptTokens, decodeTokens)
	existing := prefillScale*0.1322222222222222*prefill + decodeScale*0.28*decode
	incoming := 0.020*promptTokens + 0.060*decodeTokens
	slo := cleanSILV10TenantSLO(tenant)
	slackPressure := (existing + 0.020*incoming*(requests+1.0)) / slo

	score := reqScale*92.0*requests +
		slackScale*slackPressure +
		sameScale*14.0*sameRequests +
		sameScale*0.020*samePrefill +
		sameScale*0.090*sameDecode +
		cleanSILV10HotRequestPenalty(tenant)*excessFloat(requests, 1.0)

	if cleanSILV10IsLaneReplica(pod, tenant) {
		score -= cleanSILV10LaneRelief(tenant, promptTokens, decodeTokens)
	} else {
		score += 10.0
	}
	return score
}

func (r *cleanSILV10AgentYRouter) samplePods(pods []*v1.Pod, tenant string, requestID string) []*v1.Pod {
	if len(pods) <= 1 {
		return pods
	}
	count := cleanSILV10SampleSize(tenant)
	if count >= len(pods) {
		out := append([]*v1.Pod(nil), pods...)
		sortPodsByReplicaID(out)
		return out
	}
	type keyedPod struct {
		hash    uint32
		replica int
		pod     *v1.Pod
	}
	keyed := make([]keyedPod, 0, len(pods))
	for _, pod := range pods {
		replica := replicaID(pod)
		keyed = append(keyed, keyedPod{
			hash:    cleanSILV10HashU32(requestID, replica, cleanSILV10TenantOffset(tenant), 10091),
			replica: replica,
			pod:     pod,
		})
	}
	sort.Slice(keyed, func(i, j int) bool {
		if keyed[i].hash != keyed[j].hash {
			return keyed[i].hash < keyed[j].hash
		}
		return keyed[i].replica < keyed[j].replica
	})
	out := make([]*v1.Pod, 0, count)
	for i := 0; i < count; i++ {
		out = append(out, keyed[i].pod)
	}
	return out
}

func cleanSILV10Scales(tenant string, promptTokens, decodeTokens float64) (float64, float64, float64, float64, float64) {
	switch tenant {
	case "chat":
		return 2.20, 0.58, 0.70, 0.90, 1.55
	case "code":
		prefillScale := 1.42
		slackScale := 1.25
		if promptTokens >= 330.0 {
			prefillScale += 0.24
			slackScale += 0.18
		}
		return 0.92, prefillScale, 0.95, 0.68, slackScale
	case "research":
		decodeScale := 1.62
		slackScale := 1.18
		if decodeTokens >= 78.0 {
			decodeScale += 0.22
			slackScale += 0.12
		}
		return 0.86, 0.82, decodeScale, 1.05, slackScale
	default:
		return 1.0, 1.0, 1.0, 1.0, 1.0
	}
}

func cleanSILV10TenantSLO(tenant string) float64 {
	switch tenant {
	case "chat":
		return 0.60
	case "code":
		return 0.90
	case "research":
		return 1.05
	default:
		return 1.0
	}
}

func cleanSILV10LaneRelief(tenant string, promptTokens, decodeTokens float64) float64 {
	switch tenant {
	case "chat":
		return 3.5
	case "code":
		return 0.55 * promptTokens * 0.1322222222222222
	case "research":
		return 0.35 * decodeTokens * 0.28
	default:
		return 0.0
	}
}

func cleanSILV10HotRequestPenalty(tenant string) float64 {
	if tenant == "chat" {
		return 80.0
	}
	return 34.0
}

func cleanSILV10ShouldOverflow(laneScore, globalScore float64) bool {
	if laneScore-globalScore <= 260.0 {
		return false
	}
	if globalScore <= 0.0 {
		return true
	}
	return laneScore/globalScore >= 1.05
}

func cleanSILV10IsLaneReplica(pod *v1.Pod, tenant string) bool {
	replica := replicaID(pod)
	switch tenant {
	case "code":
		return replica == 1 || replica == 3 || replica == 5 || replica == 7
	case "research":
		return replica == 0 || replica == 1 || replica == 6 || replica == 7
	default:
		return true
	}
}

func cleanSILV10SampleSize(tenant string) int {
	return 8
}

func (r *cleanSILV10AgentYRouter) sameGPUOutstandingLocked(podName string, gpuID int) (float64, float64, float64) {
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV10AgentYRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName: pod.Name,
		tokens:  promptTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV10AgentYRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	decodeTokens := r.outputLength(ctx)
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV10AgentYRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV10AgentYRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

func cleanSILV10TieIndex(requestID string, tenant string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	tenantOffset := cleanSILV10TenantOffset(tenant)
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		return (((value + 1) * 1103515245) + 12345 + salt + tenantOffset) % count
	}
	return stableReplicaIndex(requestID+":"+tenant+":"+strconv.Itoa(salt), count)
}

func cleanSILV10TenantOffset(tenant string) int {
	switch tenant {
	case "chat":
		return 23
	case "code":
		return 59
	case "research":
		return 101
	default:
		return 0
	}
}

func cleanSILV10HashU32(requestID string, replicaID int, tenantOffset int, salt int) uint32 {
	requestValue := uint32(0)
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		requestValue = uint32(value)
	} else {
		requestValue = uint32(stableReplicaIndex(requestID, math.MaxInt32))
	}
	value := ((requestValue + 1) * 16777619) ^
		uint32((replicaID+97)*2166136261) ^
		uint32(salt) ^
		uint32(tenantOffset)
	value ^= value >> 16
	value *= 2246822519
	value ^= value >> 13
	value *= 3266489917
	value ^= value >> 16
	return value
}

var _ types.Router = (*cleanSILV10AgentYRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV10AgentYRouter)(nil)

// cleanSILV11AgentZRouter is the Go port of
// scripts/case_policies/clean_sil_v11_agent_z_router.py candidate_015. It
// keeps all tenants on a shared pool and scores each backend by outstanding
// prefill+decode tokens, softly corrected by the calibrated per-GPU runtime
// multipliers from the Layer2 profile.
type cleanSILV11AgentZRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV11AgentZRouter() (types.Router, error) {
	router := &cleanSILV11AgentZRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v11-agent-z could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV11AgentZRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)
	target, targetScore := r.bestPod(readyPods, requestID, tenant)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v11_agent_z_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV11AgentZRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV11AgentZRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV11AgentZRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV11AgentZRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV11AgentZRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV11AgentZRouter) bestPod(pods []*v1.Pod, requestID string, tenant string) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		requests int
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		scored = append(scored, scoredPod{
			score:    r.score(pod),
			requests: r.requests(pod),
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.000001 {
			break
		}
		tied = append(tied, item.pod)
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV10TieIndex(requestID, tenant, len(tied), 11017)], bestScore
}

func (r *cleanSILV11AgentZRouter) score(pod *v1.Pod) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	projectedTokens := r.pendingPrefill[pod.Name] + r.pendingDecode[pod.Name]
	return cleanSILV11EffectiveGPUMultiplier(podGPUID(pod)) * projectedTokens
}

func (r *cleanSILV11AgentZRouter) requests(pod *v1.Pod) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pendingRequests[pod.Name]
}

func cleanSILV11EffectiveGPUMultiplier(gpuID int) float64 {
	raw := cleanSILV11RawGPUMultiplier(gpuID)
	effective := 1.0 + 1.10*(raw-1.0)
	return clampFloat(effective, 0.94, 1.16)
}

func cleanSILV11RawGPUMultiplier(gpuID int) float64 {
	switch gpuID {
	case 0:
		return 1.03
	case 1:
		return 0.98
	case 2:
		return 1.12
	case 3:
		return 1.02
	default:
		return 1.0
	}
}

func (r *cleanSILV11AgentZRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV11AgentZRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV11AgentZRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV11AgentZRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

func clampFloat(value, low, high float64) float64 {
	if high < low {
		low, high = high, low
	}
	return math.Min(math.Max(value, low), high)
}

var _ types.Router = (*cleanSILV11AgentZRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV11AgentZRouter)(nil)

// cleanSILV12AgentMRouter is the exploratory Go port of
// scripts/case_policies/clean_sil_v12_agent_m_router.py candidate_014.  It
// keeps the conservative least-busy anchor from the real-validated built-in
// behavior, but adds a small gateway-visible outstanding request/token term.
// This is an exploratory real sanity check candidate, not a final E2E winner.
type cleanSILV12AgentMRouter struct {
	mu              sync.Mutex
	cache           cache.Cache
	tokenOnly       bool
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
}

func NewCleanSILV12AgentMRouter() (types.Router, error) {
	c, err := cache.Get()
	router := &cleanSILV12AgentMRouter{
		cache:           c,
		tokenOnly:       false,
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
	}
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v12-agent-m could not register request tracker: %v", err)
	}
	return router, nil
}

func NewCleanSILV12AgentMTokenRouter() (types.Router, error) {
	c, err := cache.Get()
	router := &cleanSILV12AgentMRouter{
		cache:           c,
		tokenOnly:       true,
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
	}
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v12-agent-m-token could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV12AgentMRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	target, targetScore := r.bestPod(readyPods, requestID, tenant, promptTokens, decodeTokens)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v12_agent_m_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV12AgentMRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV12AgentMRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV12AgentMRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV12AgentMRouter) SubscribedMetrics() []string {
	return []string{string(metrics.GPUBusyTimeRatio)}
}

func (r *cleanSILV12AgentMRouter) bestPod(pods []*v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		busy     float64
		requests int
		tokens   float64
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		requests, tokens := r.currentLoad(pod.Name)
		scored = append(scored, scoredPod{
			score:    r.score(pod, tenant, promptTokens, decodeTokens),
			busy:     r.busy(pod),
			requests: requests,
			tokens:   tokens,
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].busy != scored[j].busy {
			return scored[i].busy < scored[j].busy
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		if scored[i].tokens != scored[j].tokens {
			return scored[i].tokens < scored[j].tokens
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.000001 {
			break
		}
		tied = append(tied, item.pod)
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV12AgentMTieIndex(requestID, tenant, len(tied), 12011)], bestScore
}

func (r *cleanSILV12AgentMRouter) score(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	requestScale, prefillScale, decodeScale := cleanSILV12AgentMTenantScales(tenant)
	r.mu.Lock()
	prefill := r.pendingPrefill[pod.Name]
	decode := r.pendingDecode[pod.Name]
	requests := float64(r.pendingRequests[pod.Name])
	r.mu.Unlock()

	if r.tokenOnly {
		return prefill + 2.0*decode
	}
	tokenPressure := prefillScale*0.00034*prefill + decodeScale*0.00082*decode
	incomingPressure := prefillScale*0.000030*promptTokens + decodeScale*0.000070*decodeTokens
	return r.busy(pod) + requestScale*0.10*requests + tokenPressure + incomingPressure
}

func cleanSILV12AgentMTenantScales(tenant string) (float64, float64, float64) {
	const mix = 0.35
	request := 1.0
	prefill := 1.0
	decode := 1.0
	switch tenant {
	case "chat":
		request = 1.05
		prefill = 0.92
		decode = 0.94
	case "code":
		prefill = 1.04
	case "research":
		request = 0.98
		decode = 1.06
	}
	return 1.0 + mix*(request-1.0), 1.0 + mix*(prefill-1.0), 1.0 + mix*(decode-1.0)
}

func (r *cleanSILV12AgentMRouter) busy(pod *v1.Pod) float64 {
	if r.cache == nil || pod == nil {
		return 0.0
	}
	value, err := r.cache.GetMetricValueByPod(pod.Name, pod.Namespace, metrics.GPUBusyTimeRatio)
	if err != nil || value == nil {
		return 0.0
	}
	score := value.GetSimpleValue()
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0.0
	}
	return score
}

func (r *cleanSILV12AgentMRouter) currentLoad(podName string) (int, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pendingRequests[podName], r.pendingPrefill[podName] + r.pendingDecode[podName]
}

func (r *cleanSILV12AgentMRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV12AgentMRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV12AgentMRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV12AgentMRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

var _ types.Router = (*cleanSILV12AgentMRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV12AgentMRouter)(nil)

func cleanSILV12AgentMTieIndex(requestID string, tenant string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(requestID + ":" + tenant + ":" + strconv.Itoa(salt)))
	return int(h.Sum32() % uint32(count))
}

// cleanSILV12AgentLRouter is the Go port of
// scripts/case_policies/clean_sil_v12_agent_l_router.py final default.  It
// stays close to least-request routing and only reorders near-tie candidates
// using portable gateway tracker signals.
type cleanSILV12AgentLRouter struct {
	mu              sync.Mutex
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV12AgentLRouter() (types.Router, error) {
	router := &cleanSILV12AgentLRouter{
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	c, err := cache.Get()
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v12-agent-l could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV12AgentLRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)
	pool := r.nearTiePool(readyPods, tenant, promptTokens, decodeTokens)
	target, targetScore := r.bestBySecondary(pool, requestID, tenant, promptTokens, decodeTokens)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v12_agent_l_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV12AgentLRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV12AgentLRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV12AgentLRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV12AgentLRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV12AgentLRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV12AgentLRouter) nearTiePool(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64) []*v1.Pod {
	const nearTieAbs = 88.0
	const nearTieRatio = 1.09
	const maxRequestGap = 1.0
	minScore := math.MaxFloat64
	minRequests := math.MaxFloat64
	scores := make(map[string]float64, len(pods))
	requests := make(map[string]float64, len(pods))
	for _, pod := range pods {
		score := r.primaryScore(pod, tenant, promptTokens, decodeTokens)
		req := r.requests(pod.Name)
		scores[pod.Name] = score
		requests[pod.Name] = req
		if score < minScore {
			minScore = score
		}
		if req < minRequests {
			minRequests = req
		}
	}
	pool := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		if requests[pod.Name] > minRequests+maxRequestGap {
			continue
		}
		score := scores[pod.Name]
		if score-minScore <= nearTieAbs {
			pool = append(pool, pod)
			continue
		}
		if minScore > 0.0 && score/minScore <= nearTieRatio {
			pool = append(pool, pod)
		}
	}
	if len(pool) > 0 {
		return pool
	}
	best, _ := r.bestByPrimary(pods, "0", tenant, promptTokens, decodeTokens)
	if best == nil {
		return pods
	}
	return []*v1.Pod{best}
}

func (r *cleanSILV12AgentLRouter) bestByPrimary(pods []*v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		requests float64
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		scored = append(scored, scoredPod{
			score:    r.primaryScore(pod, tenant, promptTokens, decodeTokens),
			requests: r.requests(pod.Name),
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.0001 {
			break
		}
		tied = append(tied, item.pod)
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV12AgentLTieIndex(requestID, tenant, len(tied), 12071)], bestScore
}

func (r *cleanSILV12AgentLRouter) bestBySecondary(pods []*v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		primary  float64
		requests float64
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		scored = append(scored, scoredPod{
			score:    r.secondaryScore(pod, requestID, tenant, promptTokens, decodeTokens),
			primary:  r.primaryScore(pod, tenant, promptTokens, decodeTokens),
			requests: r.requests(pod.Name),
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].primary != scored[j].primary {
			return scored[i].primary < scored[j].primary
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.0001 {
			break
		}
		tied = append(tied, item.pod)
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV12AgentLTieIndex(requestID, tenant, len(tied), 12071)], bestScore
}

func (r *cleanSILV12AgentLRouter) primaryScore(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	reqScale, tokenScale, _ := cleanSILV12AgentLScales(tenant, promptTokens, decodeTokens)
	prefill, decode, requests := r.load(pod.Name)
	tokenPressure := 0.13*prefill + 0.66*decode
	incoming := 0.012*promptTokens + 0.040*decodeTokens
	return reqScale*82.0*requests + tokenScale*tokenPressure + incoming*(requests+1.0)
}

func (r *cleanSILV12AgentLRouter) secondaryScore(pod *v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) float64 {
	reqScale, tokenScale, sameScale := cleanSILV12AgentLScales(tenant, promptTokens, decodeTokens)
	rid := replicaID(pod)
	podName := pod.Name
	gpuID := podGPUID(pod)
	prefill, decode, requests := r.load(podName)
	sameReq, samePrefill, sameDecode := r.sameGPUOutstanding(podName, gpuID)
	score := reqScale*82.0*requests +
		tokenScale*(0.13*prefill+0.66*decode) +
		sameScale*18.0*sameReq +
		sameScale*0.008*samePrefill +
		sameScale*0.090*sameDecode +
		14.0*excessFloat(requests, 1.0)
	score += 0.020 * cleanSILV12AgentLHashUnit(requestID, rid, tenant, 12071)
	return score
}

func cleanSILV12AgentLScales(tenant string, promptTokens, decodeTokens float64) (float64, float64, float64) {
	req := 1.0
	token := 1.0
	same := 1.0
	switch tenant {
	case "chat":
		req = 1.22
		token = 0.62
		same = 0.70
	case "code":
		req = 0.96
		token = 1.10
		same = 0.84
	case "research":
		req = 1.18
		token = 1.60
		same = 0.92
	}
	if tenant == "code" && promptTokens >= 330.0 {
		token += 0.020
	}
	if tenant == "research" && decodeTokens >= 78.0 {
		token += 0.060
	}
	return req, token, same
}

func (r *cleanSILV12AgentLRouter) load(podName string) (float64, float64, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pendingPrefill[podName], r.pendingDecode[podName], float64(r.pendingRequests[podName])
}

func (r *cleanSILV12AgentLRouter) requests(podName string) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return float64(r.pendingRequests[podName])
}

func (r *cleanSILV12AgentLRouter) sameGPUOutstanding(podName string, gpuID int) (float64, float64, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV12AgentLRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV12AgentLRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV12AgentLRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV12AgentLRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

func cleanSILV12AgentLTieIndex(requestID string, tenant string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	tenantOffset := cleanSILV10TenantOffset(tenant)
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		return (((value + 1) * 1103515245) + 12345 + salt + tenantOffset) % count
	}
	return stableReplicaIndex(requestID+":"+tenant+":"+strconv.Itoa(salt), count)
}

func cleanSILV12AgentLHashUnit(requestID string, replicaID int, tenant string, salt int) float64 {
	requestValue := uint32(0)
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		requestValue = uint32(value)
	} else {
		requestValue = uint32(stableReplicaIndex(requestID, math.MaxInt32))
	}
	value := ((requestValue + 1) * 16777619) ^
		uint32((replicaID+97)*2166136261) ^
		uint32(salt) ^
		uint32(cleanSILV10TenantOffset(tenant))
	value ^= value >> 16
	value *= 2246822519
	value ^= value >> 13
	value *= 3266489917
	value ^= value >> 16
	return float64(value&0xFFFFFFFF) / float64(0xFFFFFFFF)
}

var _ types.Router = (*cleanSILV12AgentLRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV12AgentLRouter)(nil)

// cleanSILV13AgentNRouter is the Go port of
// scripts/case_policies/clean_sil_v13_agent_n_router.py final default.  It is
// a conservative near-tie router anchored on gateway-visible outstanding load
// rather than Vidur queue internals.
type cleanSILV13AgentNRouter struct {
	mu              sync.Mutex
	cache           cache.Cache
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV13AgentNRouter() (types.Router, error) {
	c, err := cache.Get()
	router := &cleanSILV13AgentNRouter{
		cache:           c,
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v13-agent-n could not register request tracker: %v", err)
	}
	return router, nil
}

func (r *cleanSILV13AgentNRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)
	pool := r.nearTiePool(readyPods, tenant, promptTokens, decodeTokens)
	target, targetScore := r.bestBySecondary(pool, requestID, tenant, promptTokens, decodeTokens)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v13_agent_n_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV13AgentNRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV13AgentNRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV13AgentNRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV13AgentNRouter) SubscribedMetrics() []string {
	return []string{}
}

func (r *cleanSILV13AgentNRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV13AgentNRouter) nearTiePool(pods []*v1.Pod, tenant string, promptTokens, decodeTokens float64) []*v1.Pod {
	const nearTieAbs = 72.0
	const nearTieRatio = 1.065
	const maxRequestGap = 1.0
	minScore := math.MaxFloat64
	minRequests := math.MaxFloat64
	scores := make(map[string]float64, len(pods))
	for _, pod := range pods {
		score := r.primaryScore(pod, tenant, promptTokens, decodeTokens)
		scores[pod.Name] = score
		if score < minScore {
			minScore = score
		}
		if requests := r.requests(pod.Name); requests < minRequests {
			minRequests = requests
		}
	}
	pool := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		if r.requests(pod.Name) > minRequests+maxRequestGap {
			continue
		}
		score := scores[pod.Name]
		if score-minScore <= nearTieAbs || (minScore > 0.0 && score/minScore <= nearTieRatio) {
			pool = append(pool, pod)
		}
	}
	if len(pool) > 0 {
		return pool
	}
	target, _ := r.bestByPrimary(pods, "", tenant, promptTokens, decodeTokens)
	if target == nil {
		return pods
	}
	return []*v1.Pod{target}
}

func (r *cleanSILV13AgentNRouter) bestByPrimary(pods []*v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		requests float64
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		scored = append(scored, scoredPod{
			score:    r.primaryScore(pod, tenant, promptTokens, decodeTokens),
			requests: r.requests(pod.Name),
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.0001 {
			break
		}
		tied = append(tied, item.pod)
	}
	return cleanSILV13AgentNStablePick(tied, requestID, tenant), bestScore
}

func (r *cleanSILV13AgentNRouter) bestBySecondary(pods []*v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		primary  float64
		requests float64
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		scored = append(scored, scoredPod{
			score:    r.secondaryScore(pod, requestID, tenant, promptTokens, decodeTokens),
			primary:  r.primaryScore(pod, tenant, promptTokens, decodeTokens),
			requests: r.requests(pod.Name),
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].primary != scored[j].primary {
			return scored[i].primary < scored[j].primary
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.0001 {
			break
		}
		tied = append(tied, item.pod)
	}
	return cleanSILV13AgentNStablePick(tied, requestID, tenant), bestScore
}

func (r *cleanSILV13AgentNRouter) primaryScore(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	reqScale, tokenScale, _ := cleanSILV13AgentNScales(tenant, promptTokens, decodeTokens)
	prefill, decode, requests := r.load(pod.Name)
	tokenPressure := 0.125*prefill + 0.620*decode
	incoming := 0.010*promptTokens + 0.032*decodeTokens
	score := reqScale*86.0*requests + tokenScale*tokenPressure + incoming*(requests+1.0)
	return score * cleanSILV13AgentNGPUMultiplier(podGPUID(pod))
}

func (r *cleanSILV13AgentNRouter) secondaryScore(pod *v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) float64 {
	reqScale, tokenScale, sameScale := cleanSILV13AgentNScales(tenant, promptTokens, decodeTokens)
	prefill, decode, requests := r.load(pod.Name)
	sameRequests, samePrefill, sameDecode := r.sameGPUOutstanding(pod.Name, podGPUID(pod))
	score := reqScale*86.0*requests +
		tokenScale*(0.125*prefill+0.620*decode) +
		sameScale*13.0*sameRequests +
		sameScale*0.006*samePrefill +
		sameScale*0.060*sameDecode +
		7.0*excessFloat(requests, 1.0)
	score *= cleanSILV13AgentNGPUMultiplier(podGPUID(pod))
	score += 0.001 * float64(cleanSILV13AgentNLocalReplicaID(pod))
	score += 0.012 * cleanSILV13AgentNHashUnit(requestID, replicaID(pod), tenant)
	return score
}

func cleanSILV13AgentNScales(tenant string, promptTokens, decodeTokens float64) (float64, float64, float64) {
	req := 1.0
	token := 1.0
	same := 1.0
	switch tenant {
	case "chat":
		req = 1.16
		token = 0.72
		same = 0.78
	case "code":
		req = 1.0
		token = 1.10
		same = 0.90
		if promptTokens >= 330.0 {
			token += 0.012
		}
	case "research":
		req = 1.16
		token = 1.52
		same = 1.02
		if decodeTokens >= 78.0 {
			token += 0.032
		}
	}
	return req, token, same
}

func cleanSILV13AgentNGPUMultiplier(gpuID int) float64 {
	raw := 1.0
	switch gpuID {
	case 0:
		raw = 1.03
	case 1:
		raw = 0.98
	case 2:
		raw = 1.12
	case 3:
		raw = 1.02
	}
	effective := 1.0 + 0.10*(raw-1.0)
	if effective < 0.985 {
		return 0.985
	}
	if effective > 1.025 {
		return 1.025
	}
	return effective
}

func (r *cleanSILV13AgentNRouter) load(podName string) (float64, float64, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pendingPrefill[podName], r.pendingDecode[podName], float64(r.pendingRequests[podName])
}

func (r *cleanSILV13AgentNRouter) requests(podName string) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return float64(r.pendingRequests[podName])
}

func (r *cleanSILV13AgentNRouter) sameGPUOutstanding(podName string, gpuID int) (float64, float64, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV13AgentNRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV13AgentNRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV13AgentNRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV13AgentNRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

func cleanSILV13AgentNStablePick(pods []*v1.Pod, requestID string, tenant string) *v1.Pod {
	if len(pods) == 0 {
		return nil
	}
	sortPodsByReplicaID(pods)
	if len(pods) == 1 {
		return pods[0]
	}
	return pods[cleanSILV13AgentNStableIndex(requestID, tenant, len(pods))]
}

func cleanSILV13AgentNStableIndex(requestID string, tenant string, count int) int {
	if count <= 1 {
		return 0
	}
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		index := ((value+1)*1103515245 + 12345 + 13029 + cleanSILV13AgentNTenantOffset(tenant)) % count
		if index < 0 {
			index += count
		}
		return index
	}
	return stableReplicaIndex(requestID+":"+tenant+":13029", count)
}

func cleanSILV13AgentNHashUnit(requestID string, replicaID int, tenant string) float64 {
	value := 0
	if parsed, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		value = parsed
	}
	x := uint32((value + 1) * 16777619)
	x ^= uint32((replicaID + 97) * 2166136261)
	x ^= uint32(13029)
	x ^= uint32(cleanSILV13AgentNTenantOffset(tenant))
	x ^= x >> 16
	x *= 2246822519
	x ^= x >> 13
	x *= 3266489917
	x ^= x >> 16
	return float64(x) / float64(math.MaxUint32)
}

func cleanSILV13AgentNTenantOffset(tenant string) int {
	switch tenant {
	case "chat":
		return 23
	case "code":
		return 59
	case "research":
		return 101
	default:
		return 0
	}
}

func cleanSILV13AgentNLocalReplicaID(pod *v1.Pod) int {
	if raw := pod.Labels["sil.local/local-replica-id"]; raw != "" {
		if value, err := strconv.Atoi(raw); err == nil {
			return value
		}
	}
	replica := replicaID(pod)
	if replica == math.MaxInt32 {
		return 0
	}
	return replica % 2
}

var _ types.Router = (*cleanSILV13AgentNRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV13AgentNRouter)(nil)

// cleanSILV13AgentPRouter is the Go port of
// scripts/case_policies/clean_sil_v13_agent_p_router.py default candidate.
// It keeps GPUBusyTimeRatio as the primary AIBrix-native anchor and only uses
// gateway-visible outstanding request/token state inside a narrow busy band.
type cleanSILV13AgentPRouter struct {
	mu              sync.Mutex
	cache           cache.Cache
	pendingPrefill  map[string]float64
	pendingDecode   map[string]float64
	pendingRequests map[string]int
	podGPU          map[string]int
}

func NewCleanSILV13AgentPRouter() (types.Router, error) {
	c, err := cache.Get()
	router := &cleanSILV13AgentPRouter{
		cache:           c,
		pendingPrefill:  make(map[string]float64),
		pendingDecode:   make(map[string]float64),
		pendingRequests: make(map[string]int),
		podGPU:          make(map[string]int),
	}
	if err == nil {
		c.RegisterRequestTracker(router)
	} else {
		klog.Warningf("clean-sil-v13-agent-p could not register request tracker: %v", err)
	}
	return router, nil
}

func NewCleanSILV13AgentPTightRouter() (types.Router, error) {
	return NewCleanSILV13AgentPRouter()
}

func (r *cleanSILV13AgentPRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	if len(readyPods) == 0 {
		return "", ErrorNoAvailablePod
	}

	tenant := tenantID(ctx)
	requestID := traceRequestID(ctx)
	promptTokens := r.promptLength(ctx)
	decodeTokens := r.outputLength(ctx)
	r.rememberPlacement(readyPods)
	pool := r.busyBand(readyPods)
	target, targetScore := r.bestByProjected(pool, requestID, tenant, promptTokens, decodeTokens)
	if target == nil {
		return "", ErrorNoAvailablePod
	}

	ctx.SetTargetPod(target)
	r.addPending(ctx, target, promptTokens, decodeTokens)
	klog.V(4).InfoS(
		"clean_sil_v13_agent_p_route",
		"request_id", ctx.RequestID,
		"tenant", tenant,
		"target_pod", target.Name,
		"replica_id", replicaID(target),
		"gpu_id", podGPUID(target),
		"prompt_tokens", promptTokens,
		"decode_tokens", decodeTokens,
		"score", targetScore,
	)
	return ctx.TargetAddress(), nil
}

func (r *cleanSILV13AgentPRouter) AddRequestCount(ctx *types.RoutingContext, requestID string, modelName string) int64 {
	return 0
}

func (r *cleanSILV13AgentPRouter) DoneRequestCount(ctx *types.RoutingContext, requestID string, modelName string, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV13AgentPRouter) DoneRequestTrace(ctx *types.RoutingContext, requestID string, modelName string, inputTokens, outputTokens, traceTerm int64) {
	r.done(ctx)
}

func (r *cleanSILV13AgentPRouter) SubscribedMetrics() []string {
	return []string{string(metrics.GPUBusyTimeRatio)}
}

func (r *cleanSILV13AgentPRouter) rememberPlacement(pods []*v1.Pod) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, pod := range pods {
		r.podGPU[pod.Name] = podGPUID(pod)
	}
}

func (r *cleanSILV13AgentPRouter) busyBand(pods []*v1.Pod) []*v1.Pod {
	const busyTieWindow = 0.002
	minBusy := math.MaxFloat64
	for _, pod := range pods {
		if busy := r.busy(pod); busy < minBusy {
			minBusy = busy
		}
	}
	pool := make([]*v1.Pod, 0, len(pods))
	for _, pod := range pods {
		if r.busy(pod) <= minBusy+busyTieWindow {
			pool = append(pool, pod)
		}
	}
	if len(pool) > 0 {
		return pool
	}
	return pods
}

func (r *cleanSILV13AgentPRouter) bestByProjected(pods []*v1.Pod, requestID string, tenant string, promptTokens, decodeTokens float64) (*v1.Pod, float64) {
	type scoredPod struct {
		score    float64
		busy     float64
		requests float64
		tokens   float64
		replica  int
		pod      *v1.Pod
	}
	scored := make([]scoredPod, 0, len(pods))
	for _, pod := range pods {
		prefill, decode, requests := r.load(pod.Name)
		scored = append(scored, scoredPod{
			score:    r.score(pod, tenant, promptTokens, decodeTokens),
			busy:     r.busy(pod),
			requests: requests,
			tokens:   prefill + decode,
			replica:  replicaID(pod),
			pod:      pod,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		if scored[i].busy != scored[j].busy {
			return scored[i].busy < scored[j].busy
		}
		if scored[i].requests != scored[j].requests {
			return scored[i].requests < scored[j].requests
		}
		if scored[i].tokens != scored[j].tokens {
			return scored[i].tokens < scored[j].tokens
		}
		return scored[i].replica < scored[j].replica
	})
	if len(scored) == 0 {
		return nil, 0.0
	}
	bestScore := scored[0].score
	tied := make([]*v1.Pod, 0, len(scored))
	for _, item := range scored {
		if item.score-bestScore > 0.000001 {
			break
		}
		tied = append(tied, item.pod)
	}
	sortPodsByReplicaID(tied)
	return tied[cleanSILV12AgentMTieIndex(requestID, tenant, len(tied), 13097)], bestScore
}

func (r *cleanSILV13AgentPRouter) score(pod *v1.Pod, tenant string, promptTokens, decodeTokens float64) float64 {
	requestScale, tokenScale := cleanSILV13AgentPScales(tenant)
	prefill, decode, requests := r.load(pod.Name)
	sameRequests, _, _ := r.sameGPUOutstanding(pod.Name, podGPUID(pod))
	tokenPressure := tokenScale * (0.004*prefill + 0.012*decode)
	sameGPUPressure := 0.00 * sameRequests
	return 1000.0*r.busy(pod) + requestScale*0.25*requests + tokenPressure + sameGPUPressure + 0.0*promptTokens + 0.0*decodeTokens
}

func cleanSILV13AgentPScales(tenant string) (float64, float64) {
	const mix = 0.0
	request := 1.0
	token := 1.0
	switch tenant {
	case "chat":
		token = 0.98
	case "code":
		token = 1.02
	case "research":
		token = 1.04
	}
	return 1.0 + mix*(request-1.0), 1.0 + mix*(token-1.0)
}

func (r *cleanSILV13AgentPRouter) busy(pod *v1.Pod) float64 {
	if r.cache == nil || pod == nil {
		return 1.02
	}
	value, err := r.cache.GetMetricValueByPod(pod.Name, pod.Namespace, metrics.GPUBusyTimeRatio)
	if err != nil || value == nil {
		return 1.02
	}
	score := value.GetSimpleValue()
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 1.02
	}
	return score
}

func (r *cleanSILV13AgentPRouter) load(podName string) (float64, float64, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pendingPrefill[podName], r.pendingDecode[podName], float64(r.pendingRequests[podName])
}

func (r *cleanSILV13AgentPRouter) sameGPUOutstanding(podName string, gpuID int) (float64, float64, float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	requests := 0.0
	prefill := 0.0
	decode := 0.0
	for otherPodName, otherGPU := range r.podGPU {
		if otherPodName == podName || otherGPU != gpuID {
			continue
		}
		requests += float64(r.pendingRequests[otherPodName])
		prefill += r.pendingPrefill[otherPodName]
		decode += r.pendingDecode[otherPodName]
	}
	return requests, prefill, decode
}

func (r *cleanSILV13AgentPRouter) addPending(ctx *types.RoutingContext, pod *v1.Pod, promptTokens, decodeTokens float64) {
	if ctx.Value(evolvedPartitionTrackedKey) != nil {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, evolvedPartitionTrackedRequest{
		podName:      pod.Name,
		tokens:       promptTokens,
		decodeTokens: decodeTokens,
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.podGPU[pod.Name] = podGPUID(pod)
	r.pendingPrefill[pod.Name] += promptTokens
	r.pendingDecode[pod.Name] += decodeTokens
	r.pendingRequests[pod.Name]++
}

func (r *cleanSILV13AgentPRouter) done(ctx *types.RoutingContext) {
	if ctx == nil {
		return
	}
	raw := ctx.Value(evolvedPartitionTrackedKey)
	if raw == nil {
		return
	}
	tracked, ok := raw.(evolvedPartitionTrackedRequest)
	if !ok {
		return
	}
	ctx.Context = context.WithValue(ctx.Context, evolvedPartitionTrackedKey, nil)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pendingPrefill[tracked.podName] -= tracked.tokens
	if r.pendingPrefill[tracked.podName] < 0 {
		r.pendingPrefill[tracked.podName] = 0
	}
	r.pendingDecode[tracked.podName] -= tracked.decodeTokens
	if r.pendingDecode[tracked.podName] < 0 {
		r.pendingDecode[tracked.podName] = 0
	}
	r.pendingRequests[tracked.podName]--
	if r.pendingRequests[tracked.podName] < 0 {
		r.pendingRequests[tracked.podName] = 0
	}
}

func (r *cleanSILV13AgentPRouter) promptLength(ctx *types.RoutingContext) float64 {
	if tokens, err := ctx.PromptLength(); err == nil && tokens > 0 {
		return float64(tokens)
	}
	return 1.0
}

func (r *cleanSILV13AgentPRouter) outputLength(ctx *types.RoutingContext) float64 {
	if ctx == nil || len(ctx.ReqBody) == 0 {
		return 1.0
	}
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	if json.Unmarshal(ctx.ReqBody, &body) == nil && body.MaxTokens > 0 {
		return float64(body.MaxTokens)
	}
	return 1.0
}

var _ types.Router = (*cleanSILV13AgentPRouter)(nil)
var _ cache.RequestTracker = (*cleanSILV13AgentPRouter)(nil)

func excessFloat(value, threshold float64) float64 {
	if value <= threshold {
		return 0.0
	}
	return value - threshold
}

func podGPUID(pod *v1.Pod) int {
	if pod == nil {
		return math.MaxInt32
	}
	if raw := pod.Labels["sil.local/gpu-id"]; raw != "" {
		if value, err := strconv.Atoi(raw); err == nil {
			return value
		}
	}
	replica := replicaID(pod)
	if replica == math.MaxInt32 {
		return math.MaxInt32
	}
	return replica / 2
}

func e2eLCGTieIndex(requestID string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		return (((value + 1) * 1103515245) + 12345 + salt) % count
	}
	return stableReplicaIndex(requestID+":"+strconv.Itoa(salt), count)
}

func silV6CV2TieIndex(requestID string, count int, salt int) int {
	if count <= 1 {
		return 0
	}
	if value, err := strconv.Atoi(strings.TrimSpace(requestID)); err == nil {
		return (((value + 1) * 1103515245) + salt) % count
	}
	return stableReplicaIndex(requestID+":"+strconv.Itoa(salt), count)
}

func stableReplicaIndex(requestID string, count int) int {
	if count <= 1 {
		return 0
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(requestID + ":131"))
	return int(h.Sum64() % uint64(count))
}
