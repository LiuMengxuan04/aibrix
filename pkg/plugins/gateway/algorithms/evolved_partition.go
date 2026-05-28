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
}

type evolvedPartitionTrackedRequest struct {
	podName string
	tokens  float64
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
